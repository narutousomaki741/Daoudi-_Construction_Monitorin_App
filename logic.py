import pandas as pd
import tempfile
import os, time, shutil
import datetime 
from datetime import timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import List, Dict, Optional
import bisect
import math
import warnings
import logging
import loguru
import streamlit as st
import plotly.express as px
from io import BytesIO
from utils import save_excel_file
from models import Task,BaseTask, WorkerResource, EquipmentResource
from defaults import workers, equipment, BASE_TASKS, cross_floor_links, acceleration, SHIFT_CONFIG

from helpers import (
    ResourceAllocationList,AdvancedResourceManager,EquipmentResourceManager,
    Topo_order_tasks,
    generate_tasks,
    validate_tasks,
    parse_quantity_excel,
    parse_worker_excel,
    parse_equipment_excel,
    generate_quantity_template,
    generate_worker_template,
    generate_equipment_template
)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
ground_disciplines=["Préliminaire","Terrassement","Fondations"]
# ----------------------------


# -----------------------------
# Calendar (workdays) - half-open end
# -----------------------------
class AdvancedCalendar:
    def __init__(self, start_date, holidays=None, workweek=None):
        self.current_date = pd.to_datetime(start_date)
        self.holidays = set(pd.to_datetime(h) for h in (holidays or []))
        # workweek: 0=Monday .. 6=Sunday
        self.workweek = workweek or [0, 1, 2, 3, 4]

    def is_workday(self, date):
        # date is a Timestamp or datetime
        d = pd.to_datetime(date)
        return d.weekday() in self.workweek and d.normalize() not in self.holidays

    def add_workdays(self, start_date, duration):
        """
        Returns an exclusive end date (the day after the last workday).
        Example: duration=1 -> returns start_date + 1 workday (end exclusive)
        """
        if duration <= 0:
            return pd.to_datetime(start_date)
        days = 0
        current = pd.to_datetime(start_date)
        last_workday = None
        while days < duration:
            if self.is_workday(current):
                days += 1
                last_workday = current
            current = current + timedelta(days=1)
        # return the day after the last workday (exclusive end)
        return pd.to_datetime(last_workday) + pd.Timedelta(days=1)
    def add_calendar_days(self, start_date, days):
        """Add calendar days (includes weekends/holidays) - for DELAYS"""
        """Returns exclusive end date"""
        if days <= 0:
            return pd.to_datetime(start_date)
        result = pd.to_datetime(start_date) + pd.Timedelta(days=days)
        return result

# -----------------------------
# Duration Calculator (uses separate pools)
# -----------------------------

class DurationCalculator:
    def __init__(self, workers: Dict[str, WorkerResource], equipment: Dict[str, EquipmentResource], quantity_matrix: Dict):
        self.workers = workers
        self.equipment = equipment
        self.quantity_matrix = quantity_matrix
        self.acceleration = acceleration

    def _get_quantity(self, task: Task):
        base_q = self.quantity_matrix.get(str(task.base_id), {})
        floor_q = base_q.get(task.floor, {})
    
        if not floor_q:  # Check if floor_q is empty dict or None
            print(f"⚠️ Floor {task.floor} for task {task.base_id} not found in quantity_matrix")
            qty = getattr(task, 'quantity', 1)  # Fallback
        else:
            qty = floor_q.get(task.zone, getattr(task, 'quantity', 1))
    
        if qty is None or qty <= 0:
            print(f"⚠️ Invalid quantity {qty} for task {task.base_id}, defaulting to 1")
            qty = 1
    
        print(f"✅ Task {task.base_id}, floor {task.floor} quantity: {qty}")
        task.quantity = qty
        return qty

    def _get_productivity_rate(self, resource, task_id, default=1.0):
        """Get task-specific productivity rate from resource."""
        if hasattr(resource, 'productivity_rates'):
            if isinstance(resource.productivity_rates, dict):
                return resource.productivity_rates.get(task_id, default)
            else:
                return resource.productivity_rates
        return default

    def _get_first_equipment_type(self, min_equipment_needed):
        """Get the first equipment type from min_equipment_needed dictionary."""
        if not min_equipment_needed:
            return None
        
        # Get the first key-value pair from the dictionary
        first_key = next(iter(min_equipment_needed))
        
        # If it's a tuple/list (alternative equipment), take the first one
        if isinstance(first_key, (tuple, list)):
            return first_key[0] if first_key else None
        else:
            return first_key

    def calculate_duration(self, task: Task, allocated_crews: int = None, allocated_equipments: dict = None) -> int:
        """
        Calculate workdays using the actual allocated resources:
          - allocated_crews: integer (if None -> use task.crews_needed)
          - allocated_equipment: dict {eq_name: units} (if None -> use task.equipment_needed)
        Return integer days (min 1).
          -if duration is fixed returns it
        """
        if getattr(task, "base_duration", None) is not None:
            return int(math.ceil(task.base_duration))
        qty = self._get_quantity(task)

        # normalize allocated items
        crews = allocated_crews if allocated_crews is not None else max(1, task.min_crews_needed)
        eq_alloc = allocated_equipments if allocated_equipments is not None else (task.min_equipment_needed or {})
        A=7
        if task.task_type == "worker":
            if task.resource_type not in self.workers:
                raise ValueError(f"Worker resource '{task.resource_type}' not found for task {task.id}")
            res = self.workers[task.resource_type]
            # FIXED: Use task-specific productivity rate
            base_prod = self._get_productivity_rate(res, task.base_id, 1.0)
            # worker daily production = base_prod * crews * efficiency
            daily_prod = base_prod * crews 
            if daily_prod <= 0:
                raise ValueError(f"Non-positive worker productivity for {task.id}")
            duration = qty / daily_prod
            A=base_prod
        elif task.task_type == "equipment":
            if not eq_alloc:
                raise ValueError(f"Equipment task {task.id} has no equipment specified")
            
            # FIXED: Use only the first equipment type for productivity calculation
            first_eq_type = self._get_first_equipment_type(task.min_equipment_needed)
            if not first_eq_type:
                raise ValueError(f"No equipment types found in min_equipment_needed for task {task.id}")
            
            # Get total units allocated for the first equipment type (including alternatives)
            total_units = 0
            if isinstance(first_eq_type, (tuple, list)):
                # If first equipment type is a tuple of alternatives, sum all allocated alternatives
                for eq_name in first_eq_type:
                    total_units += eq_alloc.get(eq_name, 0)
            else:
                # Single equipment type
                total_units = eq_alloc.get(first_eq_type, 0)
            
            if first_eq_type not in self.equipment:
                raise ValueError(f"First equipment type '{first_eq_type}' not found for task {task.id}")
            
            res = self.equipment[first_eq_type]
            base_prod = self._get_productivity_rate(res, task.base_id, 1.0)
            daily_prod_total = base_prod * total_units 
            A=base_prod
            if daily_prod_total <= 0:
                raise ValueError(f"Non-positive equipment productivity for {task.id}")
            duration = qty / daily_prod_total

        elif task.task_type == "hybrid":
            # worker-limited
            if task.resource_type not in self.workers:
                raise ValueError(f"Worker resource '{task.resource_type}' not found for task {task.id}")
            worker_res = self.workers[task.resource_type]
            # FIXED: Use task-specific productivity rate
            base_prod_worker = self._get_productivity_rate(worker_res, task.base_id, 1.0)
            daily_worker_prod = base_prod_worker * crews 
            if daily_worker_prod <= 0:
                raise ValueError(f"Non-positive worker productivity for {task.id}")
            A=base_prod_worker
            # FIXED: Equipment-limited using only first equipment type
            daily_equip_prod = 0
            if eq_alloc:
                first_eq_type = self._get_first_equipment_type(task.min_equipment_needed)
                if first_eq_type and first_eq_type in self.equipment:
                    total_units = 0
                    if isinstance(first_eq_type, (tuple, list)):
                        for eq_name in first_eq_type:
                            total_units += eq_alloc.get(eq_name, 0)
                    else:
                        total_units = eq_alloc.get(first_eq_type, 0)
                    
                    eq_res = self.equipment[first_eq_type]
                    base_prod_eq = self._get_productivity_rate(eq_res, task.base_id, 1.0)
                    daily_equip_prod = base_prod_eq * total_units * eq_res.efficiency
            
            # Worker duration
            duration_worker = qty / daily_worker_prod if daily_worker_prod > 0 else float('inf')
            
            # Equipment duration (if equipment is used)
            duration_equip = qty / daily_equip_prod if daily_equip_prod > 0 else float('inf')
            
            # Use the longer duration (bottleneck)
            duration = duration_worker

        else:
            raise ValueError(f"Unknown task_type: {task.task_type}")

        shift_factor = SHIFT_CONFIG.get(task.discipline, SHIFT_CONFIG.get("default", 1.0))
        duration = duration / shift_factor
        if task.floor > 1:
            duration *= 0.98 ** (task.floor - 1)

        # validate & clamp
        if not isinstance(duration, (int, float)) or math.isnan(duration) or math.isinf(duration):
            raise ValueError(f"Invalid duration for task {task.id}: {duration!r}")

        duration = float(duration)
        if duration <= 1:
            warnings.warn(f"Computed non-positive duration for task {task.id}. Setting to 1 day.")
            duration = 1.0

        duration_days = int(math.ceil(duration))
        print(f"for {task.id} duration is {duration_days} productivity is {A}")
        return max(1, duration_days)
# -----------------------------
# CPM Analyzer (day counts)
# -----------------------------


class CPMAnalyzer:
    def __init__(self, tasks, durations=None, dependencies=None):
        """
        Flexible CPM analyzer.

        Option A:
            tasks: list of Task objects
                Each Task must have .id, .predecessors, .base_duration
        Option B:
            tasks: list of task IDs
            durations: dict {task_id: duration}
            dependencies: dict {task_id: list of predecessor_ids}
        """
        # Case A: list of Task objects
        if durations is None and dependencies is None and tasks and hasattr(tasks[0], "id"):
            self.tasks = tasks
            self.task_by_id = {t.id: t for t in tasks}
            self.durations = {t.id: t.base_duration for t in tasks}
            self.dependencies = {t.id: t.predecessors for t in tasks}

        # Case B: raw IDs + dicts
        else:
            self.tasks = tasks  # here it's a list of IDs
            self.task_by_id = {tid: None for tid in tasks}  # no real Task objects
            self.durations = durations
            self.dependencies = dependencies

        # Graph data structures
        self.adj = defaultdict(list)      # successors
        self.rev_adj = defaultdict(list)  # predecessors
        self.indeg = defaultdict(int)
        self.outdeg = defaultdict(int)

        # CPM results
        self.ES, self.EF = {}, {}
        self.LS, self.LF = {}, {}
        self.float = {}
        self.project_duration = 0

    # --------------------------------------------------------
    def build_graph(self):
        """Build adjacency and degree maps."""
        for tid in self.tasks:
            preds = self.dependencies.get(tid, [])
            for p in preds:
                self.adj[p].append(tid)
                self.rev_adj[tid].append(p)
                self.indeg[tid] += 1
                self.outdeg[p] += 1

    # --------------------------------------------------------
    def forward_pass(self):
        """Compute earliest start/finish (ES/EF)."""
        q = deque([tid for tid in self.tasks if self.indeg[tid] == 0])
        while q:
            u = q.popleft()
            preds = self.dependencies.get(u, [])
            self.ES[u] = max((self.EF[p] for p in preds), default=0)
            self.EF[u] = self.ES[u] + self.durations[u]
            for v in self.adj[u]:
                self.indeg[v] -= 1
                if self.indeg[v] == 0:
                    q.append(v)
        self.project_duration = max(self.EF.values())

    # --------------------------------------------------------
    def backward_pass(self):
        """Compute latest start/finish (LS/LF)."""
        q = deque([tid for tid in self.tasks if self.outdeg[tid] == 0])
        for tid in q:
            self.LF[tid] = self.project_duration
            self.LS[tid] = self.LF[tid] - self.durations[tid]

        while q:
            u = q.popleft()
            for p in self.rev_adj[u]:
                if p not in self.LF:
                    self.LF[p] = self.LS[u]
                else:
                    self.LF[p] = min(self.LF[p], self.LS[u])
                self.LS[p] = self.LF[p] - self.durations[p]
                self.outdeg[p] -= 1
                if self.outdeg[p] == 0:
                    q.append(p)

    # --------------------------------------------------------
    def analyze(self):
        """Run full CPM analysis and calculate floats."""
        self.build_graph()
        self.forward_pass()
        self.backward_pass()

        for tid in self.tasks:
            self.float[tid] = self.LS[tid] - self.ES[tid]

        return self.project_duration

    # --------------------------------------------------------
    def get_critical_tasks(self):
        """Return list of critical task IDs (slack == 0)."""
        return [tid for tid in self.tasks if self.float.get(tid, 0) == 0]

    def get_critical_chains(self):
        """Return possible critical paths (chains)."""
        critical_paths = []

        def dfs(path):
            last = path[-1]
            extended = False
            for succ in self.adj[last]:
                if self.float.get(succ, 0) == 0:
                    dfs(path + [succ])
                    extended = True
            if not extended:
                critical_paths.append(path)

        for tid in self.tasks:
            if not self.dependencies.get(tid) and self.float.get(tid, 0) == 0:
                dfs([tid])

        return critical_paths

    def run(self):
        self.analyze()
        return self

# -----------------------------

# -----------------------------
# Advanced Scheduler
# -----------------------------
class AdvancedScheduler:
    def __init__(self, tasks: List[Task], workers: Dict[str, WorkerResource], equipment: Dict[str, EquipmentResource],
                 calendar: AdvancedCalendar, duration_calc: DurationCalculator):
        # tasks should already be Task objects; we keep only included ones
        self.tasks = [t for t in tasks if getattr(t, "included", True)]
        # map by id for quick lookup
        self.task_map = {t.id: t for t in self.tasks}
        self.workers = workers
        self.equipment = equipment
        self.calendar = calendar
        self.duration_calc = duration_calc
        self.worker_manager = AdvancedResourceManager(workers)
        self.equipment_manager = EquipmentResourceManager(equipment)

    def _all_predecessors_scheduled(self, task, schedule):
        """Return True if all predecessors (that should be considered) are in schedule."""
        for p in task.predecessors:
            # If predecessor doesn't exist in our task_map, it's an error in generation (should be caught earlier)
            if p not in self.task_map:
                raise ValueError(f"Task {task.id} has a predecessor {p} that is not part of the task set.")
            if p not in schedule:
                return False
        return True

    def _earliest_start_from_preds(self, task, schedule):
        """Compute earliest possible start (exclusive end of predecessors)."""
        pred_end_dates = [
            self.calendar.add_calendar_days(schedule[p][1], self.task_map[p].delay)
            for p in task.predecessors
            if p in schedule and schedule[p][1] is not None
        ]
        if not pred_end_dates:
            print(f"[DEBUG earliest_start] Task {task.id}: No scheduled predecessors found at this moment")
            print(f"  Predecessors: {task.predecessors}")
            print(f"  Schedule keys currently: {list(schedule.keys())}")
        if pred_end_dates:
            return max(pred_end_dates)
        else:
            return self.calendar.current_date

    def _allocate_resources_for_window(self, task, start_date, duration_days):
        """Helper method to allocate resources for a specific time window."""
        end_date = self.calendar.add_workdays(start_date, duration_days)
        
        # Release any previous allocations for this task
        self.worker_manager.release(task.id)
        self.equipment_manager.release(task.id)
        
        # Compute allocations
        possible_crews = None
        if task.task_type in ("worker", "hybrid"):
            possible_crews = self.worker_manager.compute_allocation(task, start_date, end_date)

        possible_equip = {}
        if task.task_type in ("equipment", "hybrid") and (task.min_equipment_needed or {}):
            possible_equip = self.equipment_manager.compute_allocation(task, start_date, end_date) or {}

        return possible_crews, possible_equip, end_date

    def _check_feasibility(self, task, possible_crews, possible_equip):
        """Check if allocated resources meet minimum requirements."""
        min_crews = getattr(task, "min_crews_needed", max(1, task.min_crews_needed))
        
        feasible_workers = True
        if task.task_type in ("worker", "hybrid"):
            feasible_workers = (possible_crews is not None and possible_crews >= min_crews)

        feasible_equip = True
        if task.task_type in ("equipment", "hybrid") and (task.min_equipment_needed or {}):
            for eq_key, min_req in task.min_equipment_needed.items():
                eq_choices = eq_key if isinstance(eq_key, (tuple, list)) else (eq_key,)
                allocated_total = sum(possible_equip.get(eq, 0) for eq in eq_choices)
                if allocated_total < min_req:
                    feasible_equip = False
                    break

        return feasible_workers, feasible_equip

    def generate(self):
        schedule = {}
        unscheduled = set(self.task_map.keys())

        # Predecessor counts for ready queue
        pred_count = {tid: len(self.task_map[tid].predecessors) for tid in self.task_map}

        # Start with zero-predecessor tasks
        ready = deque([tid for tid, cnt in pred_count.items() if cnt == 0])

        # Validate references early
        for tid, t in self.task_map.items():
            for p in t.predecessors:
                if p not in self.task_map:
                    raise ValueError(f"Task {tid} references predecessor {p} which does not exist.")
        
        # Precompute durations (fail early)
        for tid, t in self.task_map.items():
            try:
                # compute a nominal duration using nominal resources (no allocations)
                d = self.duration_calc.calculate_duration(t)
                if not isinstance(d, int) or d < 0:
                    raise ValueError(f"Computed invalid duration {d!r}")
            except Exception as e:
                print(f"[DUR ERROR] Task {tid}: cannot compute nominal duration before scheduling => {e!r}")
                raise
            # store precomputed nominal duration on task so scheduler can use it as a fallback
            t.nominal_duration = d

        max_attempts = len(self.task_map) * 10 + 1000
        attempts = 0

        while unscheduled:
            if not ready:
                pending = ", ".join(sorted(unscheduled))
                raise RuntimeError(f"No tasks are ready but unscheduled remain: {pending}")

            tid = ready.popleft()
            task = self.task_map[tid]

            # All predecessors must be scheduled
            if not self._all_predecessors_scheduled(task, schedule):
                ready.append(tid)
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError("Scheduler stuck waiting for predecessors.")
                continue

            # Earliest start after predecessors
            start_date = self._earliest_start_from_preds(task, schedule)
            task.earliest_start = start_date

            # Initial duration guess (using nominal crews/equipment)
            duration_days = self.duration_calc.calculate_duration(task)
            if not isinstance(duration_days, int) or duration_days < 0:
                raise ValueError(f"Invalid duration for {task.id}: {duration_days}")

            # Handle instantaneous tasks
            if duration_days == 0:
                schedule[task.id] = (start_date, start_date)
                unscheduled.remove(task.id)
                task.allocated_crews = 0
                task.allocated_equipments = {}
                
                # Update successors
                for succ in [s for s in self.task_map if task.id in self.task_map[s].predecessors]:
                    pred_count[succ] -= 1
                    if pred_count[succ] == 0:
                        ready.append(succ)
                continue

            # Resource allocation loop
            allocated_crews = None
            allocated_equipments = None
            forward_attempts = 0
            max_forward = 3000

            while forward_attempts < max_forward:
                # Get resource allocations for current window
                possible_crews, possible_equip, end_date = self._allocate_resources_for_window(
                    task, start_date, duration_days
                )

                # Check feasibility
                feasible_workers, feasible_equip = self._check_feasibility(task, possible_crews, possible_equip)

                # If feasible, calculate duration with actual allocations
                if feasible_workers and feasible_equip:
                    actual_duration = self.duration_calc.calculate_duration(
                        task,
                        allocated_crews=possible_crews,
                        allocated_equipments=possible_equip
                    )
                    
                    # Re-check allocations with actual duration
                    final_crews, final_equip, final_end = self._allocate_resources_for_window(
                        task, start_date, actual_duration
                    )
                    
                    final_feasible_workers, final_feasible_equip = self._check_feasibility(task, final_crews, final_equip)
                    
                    if final_feasible_workers and final_feasible_equip:
                        # Commit allocations
                        allocated_crews = final_crews
                        allocated_equipments = final_equip

                        if allocated_crews:
                            self.worker_manager.allocate(task, start_date, final_end, allocated_crews)
                        if allocated_equipments:
                            self.equipment_manager.allocate(task, start_date, final_end, allocated_equipments)

                        duration_days = actual_duration
                        end_date = final_end
                        break

                # Shift window forward
                start_date = self.calendar.add_workdays(start_date, 1)
                forward_attempts += 1

            if forward_attempts >= max_forward:
                raise RuntimeError(f"Could not find resource window for task {task.id} after {max_forward} attempts.")

            # Final dependency enforcement
            for p in task.predecessors:
                pred_end = schedule[p][1]
                if start_date < pred_end:
                    raise RuntimeError(
                        f"Dependency violation: Task {task.id} starts {start_date} before predecessor {p} ends {pred_end}"
                    )

            # Record schedule
            schedule[task.id] = (start_date, end_date)
            unscheduled.remove(task.id)
            task.allocated_crews = allocated_crews
            task.allocated_equipments = allocated_equipments
            
            # Update successors
            for succ in [s for s in self.task_map if task.id in self.task_map[s].predecessors]:
                pred_count[succ] -= 1
                if pred_count[succ] == 0:
                    ready.append(succ)

        return schedule

# -----------------------------
# Reporter (exports Excel)
# -----------------------------

# -----------------------------
# Task generator (zones/floors expansion)
# -----------------------------


    
def run_schedule(zone_floors, quantity_matrix, start_date, workers_dict=None, equipment_dict=None, holidays=None):
    """
    Run the scheduling logic using either default dictionaries or uploaded user data.
    Returns schedule (dict of DataFrames) and output folder path.
    """
    from reporting import BasicReporter
    # Use defaults if no user input
    workers_used = workers_dict if workers_dict else workers
    equipment_used = equipment_dict if equipment_dict else equipment

    # Generate tasks
    tasks = generate_tasks(BASE_TASKS, zone_floors, cross_floor_links=cross_floor_links)

    # Validate tasks and patch missing data
    tasks, workers_used, equipment_used, quantity_matrix = validate_tasks(tasks, workers_used, equipment_used, quantity_matrix)

    # Calendar and duration
    workweek = [0, 1, 2, 3, 4, 5]
    start_date= pd.Timestamp(start_date)
    cal = AdvancedCalendar(start_date=start_date, holidays=holidays, workweek=workweek)
    dur_calc = DurationCalculator(workers_used, equipment_used, quantity_matrix)

    # Scheduler
    sched = AdvancedScheduler(tasks, workers_used, equipment_used, cal, dur_calc)
    schedule = sched.generate()

    # Reporting
    reporter = BasicReporter(tasks, schedule, sched.worker_manager, sched.equipment_manager, cal)
    output_folder = reporter.export_all()

    return schedule, output_folder

# ---------------- Final generate_schedule_ui ----------------

def generate_schedule_ui():
    """Main Streamlit interface for Construction Scheduling Web App"""
  
    from reporting import generate_interactive_gantt
    # Page setup
    st.set_page_config(layout="wide", page_title="🏗️ Construction Scheduler")
    st.header("🏗️ Construction Project Scheduler")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Project Setup", "📁 Templates", "📤 Upload Data", "🚀 Generate & Results"]
    )

    # ---------------- TAB 1: PROJECT SETUP ----------------
    with tab1:
        st.subheader("Project Configuration")

        with st.expander("🏗️ Building Configuration", expanded=True):
            num_zones = st.number_input(
                "How many zones does your building have?",
                min_value=1,
                max_value=30,
                value=2,
                help="Each zone is a distinct section of your building",
            )

            # Removed "Total Floors" input globally; only per-zone floor counts remain
            zones_floors = {}
            for i in range(num_zones):
                st.markdown(f"**Zone {i + 1}**")
                col1, col2 = st.columns([2, 1])
                with col1:
                    zone_name = st.text_input(
                        "Zone name",
                        value=f"Zone_{i + 1}",
                        key=f"zone_{i}",
                        placeholder="e.g., North_Wing",
                    )
                with col2:
                    floors = st.number_input(
                        "Floors", min_value=1, max_value=60, value=5, key=f"floors_{i}"
                    )
                zones_floors[zone_name] = floors

        start_date = st.date_input("Project Start Date", value=pd.Timestamp.today())

        with st.expander("ℹ️ Project Information", expanded=False):
            project_name = st.text_input("Project Name", value="My Construction Project")
            project_manager = st.text_input("Project Manager", placeholder="Enter project manager name")

    # ---------------- TAB 2: TEMPLATE DOWNLOADS ----------------
    with tab2:
        st.subheader("📊 Generate Default Data Templates")

        st.markdown("""
        **Step 1:** Download and fill templates with your project data:  
        - Quantity Template → task quantities per zone/floor  
        - Worker Template → crew sizes and productivity rates  
        - Equipment Template → machine counts and rates
        """)

        if "templates_ready" not in st.session_state:
            st.session_state["templates_ready"] = False

        # Unified single download action
        if st.button("📥 Generate Templates", type="primary", use_container_width=True):
            try:
                with st.spinner("Preparing templates..."):
                    qty_file = generate_quantity_template(BASE_TASKS, zones_floors)
                    worker_file = generate_worker_template(workers)
                    equip_file = generate_equipment_template(equipment)

                    st.session_state["templates_ready"] = True
                    st.session_state["qty_file"] = qty_file
                    st.session_state["worker_file"] = worker_file
                    st.session_state["equip_file"] = equip_file

                st.success("✅ Templates generated successfully!")
            except Exception as e:
                st.error(f"❌ Failed to generate templates: {e}")

        # Persistent download buttons after generation
        if st.session_state.get("templates_ready", False):
            st.markdown("### ⬇️ Download Templates")
            col1, col2, col3 = st.columns(3)
            for label, key, col in [
                ("Quantity Template", "qty_file", col1),
                ("Worker Template", "worker_file", col2),
                ("Equipment Template", "equip_file", col3),
            ]:
                path = st.session_state.get(key)
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        col.download_button(
                            f"📄 {label}",
                            f,
                            file_name=os.path.basename(path),
                            use_container_width=True,
                        )
    # ---------------- TAB 3: UPLOAD ----------------
    with tab3:
        st.subheader("📤 Upload Your Data")

        def upload_section(title, key_suffix):
            st.markdown(f"**{title}**")
            uploaded = st.file_uploader(
                f"Upload {title}", type=["xlsx"], key=f"upload_{key_suffix}"
            )
            if uploaded:
                size_kb = uploaded.size / 1024
                st.success(f"✅ {uploaded.name} uploaded ({size_kb:.1f} KB)")
                if size_kb < 5:
                    st.warning("⚠️ File seems small, please verify contents.")
            else:
                st.info(f"📄 Awaiting {title} upload...")
            return uploaded

        quantity_file = upload_section("Quantity Matrix", "quantity")
        worker_file = upload_section("Worker Template", "worker")
        equipment_file = upload_section("Equipment Template", "equipment")

    # ---------------- TAB 4: GENERATE SCHEDULE ----------------
    with tab4:
        st.subheader("🚀 Generate Schedule")

        all_ready = all([quantity_file, worker_file, equipment_file])
        if all_ready:
            st.success("✅ All files uploaded and ready!")
        else:
            missing = [
                name
                for name, file in [
                    ("Quantity Matrix", quantity_file),
                    ("Worker Template", worker_file),
                    ("Equipment Template", equipment_file),
                ]
                if not file
            ]
            st.warning(f"📋 Missing: {', '.join(missing)}")

        if st.button("🚀 Generate Project Schedule", type="primary", use_container_width=True, disabled=not all_ready):
            try:
                progress = st.progress(0)
                status = st.empty()

                # --- Parse input files ---
                status.subheader("📊 Parsing Excel files...")
                df_quantity = pd.read_excel(quantity_file)
                quantity_used = parse_quantity_excel(df_quantity)
                progress.progress(25)

                df_worker = pd.read_excel(worker_file)
                workers_used = parse_worker_excel(df_worker)
                progress.progress(45)

                df_equip = pd.read_excel(equipment_file)
                equipment_used = parse_equipment_excel(df_equip)
                progress.progress(60)

                # --- Run scheduling logic ---
                status.subheader("🏗️ Running Schedule Engine...")
                schedule, output_folder = run_schedule(
                    zone_floors=zones_floors,
                    quantity_matrix=quantity_used,
                    start_date=start_date,
                    workers_dict=workers_used or workers,
                    equipment_dict=equipment_used or equipment,
                    holidays=None,
                )
                progress.progress(90)
                time.sleep(0.5)

                # --- Handle schedule output ---
                status.subheader("💫 Finalizing Output...")
                progress.progress(100)
                st.success("🎉 Schedule generated successfully!")

                # --- Merge all schedule parts safely ---
                if isinstance(schedule, dict):
                    all_tasks = pd.concat(
                        [v for v in schedule.values() if isinstance(v, pd.DataFrame)],
                        ignore_index=True
                    )
                else:
                    all_tasks = schedule
                start_ts = pd.Timestamp(start_date)
                project_end = pd.to_datetime(all_tasks["End"].max())
                project_end=pd.Timestamp(project_end)
                duration_days = (project_end - start_ts).days

                # --- Summary ---
                st.subheader("📊 Project Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Tasks", len(all_tasks))
                c2.metric("Project Duration", f"{duration_days} days", f"Ends {project_end:%b %d, %Y}")
                c3.metric("Avg Task Duration", f"{duration_days / max(1, len(all_tasks)):.1f} days")
                c4.metric("Zones", len(zones_floors))

                # --- Download results ---
                if os.path.exists(output_folder):
                    st.subheader("📂 Download Results")
                    zip_path = os.path.join(
                        output_folder,
                        f"project_schedule_{start_date.strftime('%Y%m%d')}.zip",
                    )
                    shutil.make_archive(zip_path[:-4], "zip", output_folder)
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            "📦 Download Full Report Package",
                            f,
                            file_name=os.path.basename(zip_path),
                            mime="application/zip",
                        )
                # --- Gantt chart generation ---
                st.subheader("📊 Interactive Gantt Chart")
                schedule_excel = next(
                    (
                        os.path.join(output_folder, f)
                        for f in os.listdir(output_folder)
                        if "schedule" in f.lower() and f.endswith(".xlsx")
                    ),
                    None,
                )
                if schedule_excel and os.path.exists(schedule_excel):
                    df = pd.read_excel(schedule_excel)
                    gantt_html = os.path.join(output_folder, "interactive_gantt.html")
                    generate_interactive_gantt(df, gantt_html)

                    with open(gantt_html, "rb") as f:
                        st.download_button(
                            "📊 Download Interactive Gantt",
                            f,
                            file_name="interactive_gantt.html",
                            mime="text/html",
                        )
                    st.success("✅ Interactive Gantt generated!")
                else:
                    st.warning("⚠️ No schedule file found for Gantt chart generation.")
            except Exception as e:
                st.error(f"❌ Failed to generate schedule: {e}")
                if st.checkbox("🔍 Show error details"):
                    st.exception(e)
    # ---------------- SIDEBAR HELP ----------------
    with st.sidebar:
        st.header("💡 Help & Guidance")
        st.markdown("""
        **Steps:**
        1️⃣ Configure project zones & floors  
        2️⃣ Generate Excel templates  
        3️⃣ Upload filled data  
        4️⃣ Generate optimized schedule  

        **Required Files:**
        - Quantity Matrix  
        - Worker Template  
        - Equipment Template
        """)
def analyze_project_progress(reference_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute planned vs actual progress time series and deviations.
    Returns a DataFrame indexed by Date with PlannedProgress, Progress (daily average),
    CumulativeActual and ProgressDeviation columns.

    This version is robust to:
      - missing sheets/columns,
      - empty uploaded files,
      - different date formats,
      - missing Progress column (treat as 0),
      - gaps in dates (forward-fill).
    """
    # defensive copies
    ref_df = reference_df.copy()
    act_df = actual_df.copy()

    # Ensure expected columns exist in reference schedule
    # Ideally the schedule sheet has columns Start, End, TaskID (or TaskName).
    for col in ("Start", "End"):
        if col not in ref_df.columns:
            raise ValueError(f"Reference schedule missing required column '{col}'")

    # Parse dates robustly
    ref_df["Start"] = pd.to_datetime(ref_df["Start"], errors="coerce")
    ref_df["End"] = pd.to_datetime(ref_df["End"], errors="coerce")
    if ref_df["Start"].isna().all() or ref_df["End"].isna().all():
        raise ValueError("Reference schedule dates could not be parsed. Check Start/End columns.")

    # Build timeline (daily)
    timeline_start = ref_df["Start"].min()
    timeline_end = ref_df["End"].max()
    if pd.isna(timeline_start) or pd.isna(timeline_end):
        raise ValueError("Reference schedule dates invalid (start/end).")

    timeline = pd.date_range(timeline_start.normalize(), timeline_end.normalize(), freq="D")

    # Planned curve: fraction of tasks active on each day
    planned_curve = []
    total_tasks = max(1, len(ref_df))  # avoid division by zero
    for day in timeline:
        ongoing = ref_df[(ref_df["Start"].dt.normalize() <= day) & (ref_df["End"].dt.normalize() >= day)]
        planned_progress = len(ongoing) / total_tasks
        planned_curve.append({"Date": day, "PlannedProgress": planned_progress})

    planned_df = pd.DataFrame(planned_curve)
    planned_df["Date"] = pd.to_datetime(planned_df["Date"])
    planned_df = planned_df.set_index("Date")

    # Actual progress: expect actual_df to have Date and Progress columns
    if "Date" not in act_df.columns:
        # No actual progress provided — return planned_df with NaNs for actual
        planned_df = planned_df.reset_index()
        planned_df["Progress"] = 0.0
        planned_df["CumulativeActual"] = planned_df["Progress"].cumsum().clip(upper=1.0)
        planned_df["ProgressDeviation"] = planned_df["CumulativeActual"] - planned_df["PlannedProgress"]
        return planned_df

    # Parse actual dates; handle missing or malformed Progress
    act_df["Date"] = pd.to_datetime(act_df["Date"], errors="coerce")
    act_df = act_df.dropna(subset=["Date"])
    if act_df.empty:
        # treat as no progress recorded
        planned_df = planned_df.reset_index()
        planned_df["Progress"] = 0.0
        planned_df["CumulativeActual"] = planned_df["Progress"].cumsum().clip(upper=1.0)
        planned_df["ProgressDeviation"] = planned_df["CumulativeActual"] - planned_df["PlannedProgress"]
        return planned_df

    if "Progress" not in act_df.columns:
        # maybe user provided percent column named differently — try a few guesses
        candidate = None
        for c in ("Pct", "Percentage", "Percent", "Value"):
            if c in act_df.columns:
                candidate = c; break
        if candidate:
            act_df["Progress"] = pd.to_numeric(act_df[candidate], errors="coerce").fillna(0.0)
        else:
            act_df["Progress"] = 0.0
    else:
        act_df["Progress"] = pd.to_numeric(act_df["Progress"], errors="coerce").fillna(0.0)

    # Aggregate actual progress by Date (mean)
    actual_daily = act_df.groupby(act_df["Date"].dt.normalize(), as_index=True).agg({"Progress": "mean"})
    actual_daily.index.name = "Date"

    # Reindex to planned timeline with forward-fill/backfill as appropriate
    full_index = pd.DatetimeIndex(timeline)
    actual_daily = actual_daily.reindex(full_index, method=None)  # allow NaNs
    actual_daily["Progress"] = actual_daily["Progress"].fillna(0.0)  # if no measurement => 0 progress that day

    # Cumulative actual progress
    actual_daily["CumulativeActual"] = actual_daily["Progress"].cumsum()
    actual_daily["CumulativeActual"] = actual_daily["CumulativeActual"].clip(upper=1.0)

    # Combine planned and actual
    combined = pd.DataFrame(index=full_index)
    combined["PlannedProgress"] = planned_df["PlannedProgress"].reindex(full_index, fill_value=0.0)
    combined["Progress"] = actual_daily["Progress"]
    combined["CumulativeActual"] = actual_daily["CumulativeActual"]
    combined["ProgressDeviation"] = combined["CumulativeActual"] - combined["PlannedProgress"]
    combined = combined.reset_index().rename(columns={"index": "Date"})
    return combined
def monitor_project_ui():
    """
    Streamlit UI for project monitoring. Only runs analysis when both files are present.
    - reference_file: Reference schedule Excel (sheet 'Schedule' or a sheet having Start/End)
    - actual_file: Actual progress Excel (must contain Date and Progress)
    """
    st.header("📊 Project Monitoring (S-Curve & Deviation)")
    st.markdown(
        "Upload a **Reference Schedule** (Excel with a 'Schedule' sheet containing Start/End) "
        "and an **Actual Progress** file (Date, Progress). Analysis runs only when both are uploaded."
    )

    reference_file = st.file_uploader("Upload Reference Schedule Excel (.xlsx) — the generated schedule (sheet 'Schedule')", type=["xlsx"], key="ref_schedule")
    actual_file = st.file_uploader("Upload Actual Progress Excel (.xlsx) — rows with Date and Progress (0-1 or 0-100)", type=["xlsx"], key="actual_progress")

    # show quick-help / sample templates
    with st.expander("Help: expected formats / sample rows"):
        st.markdown("""
        **Reference schedule** — must contain `Start` and `End` columns (dates).  
        Example sheet 'Schedule' created by the generator.  
        **Actual progress** — should contain `Date` (date) and `Progress` (float 0-1 or 0-100).  
        If Progress is 0-100, it will be normalized to 0-1.
        """)

    # If user uploaded the reference file only, show schedule preview and allow download
    if reference_file and not actual_file:
        try:
            # try sheet named 'Schedule' first, otherwise first sheet
            try:
                ref_df = pd.read_excel(reference_file, sheet_name="Schedule")
            except Exception:
                reference_file.seek(0)
                ref_df = pd.read_excel(reference_file)
            st.subheader("Reference schedule preview")
            st.dataframe(ref_df.head(200))
            st.info("Upload an 'Actual Progress' file to perform monitoring analysis.")
        except Exception as e:
            st.error(f"Unable to read reference schedule: {e}")
        return

    # If both files present, compute analysis
    if reference_file and actual_file:
        try:
            # Prefer sheet 'Schedule' for the reference file
            try:
                reference_file.seek(0)
                ref_df = pd.read_excel(reference_file, sheet_name="Schedule")
            except Exception:
                reference_file.seek(0)
                ref_df = pd.read_excel(reference_file)

            actual_file.seek(0)
            act_df = pd.read_excel(actual_file)

            # Normalize 'Progress' if expressed 0-100 to 0-1
            if "Progress" in act_df.columns:
                max_val = act_df["Progress"].max(skipna=True)
                if max_val is not None and max_val > 1.1:
                    act_df["Progress"] = act_df["Progress"] / 100.0

            # Use MonitoringReporter from reporting.py if available (preferred)
            try:
                from reporting import MonitoringReporter
                reporter = MonitoringReporter(ref_df, act_df)
                # If class implements compute_analysis() and plotting helpers:
                if hasattr(reporter, "compute_analysis"):
                    reporter.compute_analysis()
                    analysis_df = getattr(reporter, "analysis_df", None)
                    if analysis_df is None:
                        # fallback to local analyzer
                        analysis_df = analyze_project_progress(ref_df, act_df)
                else:
                    analysis_df = analyze_project_progress(ref_df, act_df)
            except Exception:
                # fallback if import/class fails
                analysis_df = analyze_project_progress(ref_df, act_df)

            # Show S-curve and deviation
            st.subheader("📈 S-Curve (Planned vs Actual cumulative progress)")
            if analysis_df.empty:
                st.warning("No data produced by analysis.")
            else:
                # Using plotly express for a clean S-curve
                import plotly.express as px
                fig_s = px.line(analysis_df, x="Date", y=["PlannedProgress", "CumulativeActual"],
                                labels={"value": "Cumulative Progress", "variable": "Series"},
                                title="S-Curve: Planned vs Actual")
                st.plotly_chart(fig_s, use_container_width=True)

                st.subheader("📊 Deviation (Actual - Planned)")
                fig_dev = px.area(analysis_df, x="Date", y="ProgressDeviation",
                                  title="Progress Deviation (Actual - Planned)")
                st.plotly_chart(fig_dev, use_container_width=True)

            # provide analysis csv download
            csv_bytes = analysis_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download analysis CSV", csv_bytes, file_name="monitoring_analysis.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Monitoring analysis failed: {e}")
            import traceback, sys
            tb = traceback.format_exc()
            st.code(tb)
        return

    # If neither file provided
    if not reference_file and not actual_file:
        st.info("Upload files to start monitoring. For schedule generation use the Generate Schedule tab.")
        
