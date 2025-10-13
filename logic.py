import pandas as pd
import tempfile
import os
from datetime import timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import List, Dict, Optional
import bisect
import math
import warnings
from datetime import datetime
import logging
import loguru
import streamlit as st
import plotly.express as px
from io import BytesIO
from utils import save_excel_file
from models import Task,BaseTask, WorkerResource, EquipmentResource
from defaults import workers, equipment, BASE_TASKS, cross_floor_links, acceleration, SHIFT_CONFIG

from helpers import (
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


class ResourceAllocationList:
    def __init__(self):
        self.intervals = []  # sorted list of (start, end)

    def is_free(self, start, end):
        i = bisect.bisect_left(self.intervals, (start, end))
        if i > 0 and self.intervals[i-1][1] > start:
            return False
        if i < len(self.intervals) and self.intervals[i][0] < end:
            return False
        return True

    def add(self, start, end):
        bisect.insort(self.intervals, (start, end))

# -----------------------------
# Resource Managers
# -----------------------------G


class AdvancedResourceManager:
    """
    Manages worker crews with flexible allocation:
      - If full crews available -> allocate full crews_needed (after acceleration/clamps)
      - If not, allocate the maximum possible >= min_crews_needed
      - If less than min_crews_needed -> allocation fails (returns 0)
    Allocations stored as: allocations[res_name] -> list of (task_id, resource_name, units, start, end)
    """

    def __init__(self, workers: Dict[str, WorkerResource]):
        self.workers = workers
        self.allocations = defaultdict(list)

    def _used_crews(self, res_name, start, end):
        """Sum units already reserved overlapping [start, end)"""
        used = 0
        for (_tid, _rname, units, s, e) in self.allocations[res_name]:
            # overlap if not (end <= s or start >= e)
            if not (end <= s or start >= e):
                used += units
        return used

    def compute_allocation(self, task, start, end):
        """
        Flexible allocation policy.
        Returns integer number of crews to allocate (>= min_crews_needed), or 0 if cannot satisfy minimum.
        """
        if task.task_type == "equipment":
            return 0  # worker manager not responsible

        res_name = task.resource_type
        if res_name not in self.workers:
            return 0

        res = self.workers[res_name]
        total_pool = int(res.count)

        # find already-used crews in the window
        used = self._used_crews(res_name, start, end)
        available = max(0, total_pool - used)

        min_needed = max(1, int(task.min_crews_needed))

        # acceleration config may increase desired crews (factor) but we cap by task max and pool limits
        acc = acceleration.get(
            task.discipline,
            acceleration.get("default", {"factor": 1.0})
        )
        factor = acc.get("factor", 1.0)

        # ideal after acceleration (but must be <= disc_max and <= per-res max)
        candidate = int(math.ceil(min_needed * factor))
        
        # FIXED: Handle max_crews properly for both dict and legacy int types
        res_max_value = getattr(res, "max_crews", None)
        
        if isinstance(res_max_value, dict):
            # Dictionary case: get task-specific limit
            task_id = getattr(task, 'id', None)
            if task_id and task_id in res_max_value:
                task_max = res_max_value[task_id]
                candidate = min(candidate, int(task_max))
                print(f"[ALLOC DEBUG] Using task-specific max_crews: {task_max} for task {task_id}")
        elif res_max_value is not None :
            if res_max_value > 0:
            # Legacy single integer case
                candidate = min(candidate, int(res_max_value))
        
        print(f"[ALLOC DEBUG] {task.id} disc={task.discipline} min_needed={min_needed} "
              f"factor={factor} candidate={candidate} pool={total_pool} used={used}")
        
        # final allocation is the maximum we can give within [min_needed, candidate] limited by available
        allocated = min(candidate, available)

        # If allocated is less than minimum, fail
        if allocated < min_needed:
            print(f"[ALLOC FAIL] {task.id} pool={total_pool} used={used} available={available} min_needed={min_needed} candidate={candidate}")
            return 0

        return int(allocated)

    def can_allocate(self, task, start, end):
        alloc = self.compute_allocation(task, start, end)
        return alloc >= max(1, getattr(task, "min_crews_needed", max(1, task.crews_needed)))

    def allocate(self, task, start, end, units):
        """
        Reserve exactly `units` crews for this task in [start, end).
        Returns units reserved or 0 on failure.
        """
        if units is None or units <= 0:
            return 0
        # append allocation record
        self.allocations[task.resource_type].append((task.id, task.resource_type, int(units), start, end))
        return int(units)

    def release(self, task_id):
        """Release all allocations associated with a task id."""
        for res_name in list(self.allocations.keys()):
            self.allocations[res_name] = [a for a in self.allocations[res_name] if a[0] != task_id]
# -----------------------------
# Equipment Manager (shared use)
# -----------------------------

class EquipmentResourceManager:
    """
    Professional equipment allocation manager.

    Features:
    - Split allocation across alternative equipment types
    - Respect individual equipment maximums (max_equipment)
    - Multi-phase allocation: min_required → accelerated_target
    - Cost-aware optimization with weighted strategy
    - Allocations stored as: allocations[equipment_name] -> list of (task_id, equipment_name, units, start, end)
    """

    def __init__(self, equipment: dict):
        self.equipment = equipment
        self.allocations = defaultdict(list)

    def allocate(self, task, start, end, allocation: dict = None):
        """
        Reserve the explicit allocation dict {eq_name: units} for this task.
        If allocation is None, compute automatically.
        """
        if allocation is None:
            allocation = self.compute_allocation(task, start, end)
        if not allocation:
            return None
        for eq_name, units in allocation.items():
            self.allocations[eq_name].append((task.id, eq_name, int(units), start, end))
        return allocation

    def _used_units(self, eq_name, start, end):
        """Sum units already reserved overlapping [start, end)."""
        used = 0
        for (_tid, _, units, s, e) in self.allocations[eq_name]:
            if not (end <= s or start >= e):
                used += units
        return used

    def compute_allocation(self, task, start, end):
        """
        Advanced equipment allocation with professional multi-equipment support.

        Returns: {equipment_name: allocated_units} or None if requirements cannot be met
        """
        if not task.min_equipment_needed:
            return {}

        final_allocation = {}

        for eq_key, requested_units in task.min_equipment_needed.items():
            eq_choices = self._normalize_equipment_choices(eq_key)

            # Phase 1: Calculate requirements with acceleration
            min_required = int(requested_units)
            target_demand = self._calculate_accelerated_demand(min_required, task.discipline)

            # Phase 2: Analyze equipment availability
            equipment_analysis = self._analyze_equipment_availability(eq_choices, start, end, target_demand, task)
            if not equipment_analysis:
                self._log_allocation_failure(task, eq_choices, min_required, equipment_analysis)
                return None

            # Phase 3: Multi-stage allocation
            allocation_result = self._perform_multi_stage_allocation(equipment_analysis, min_required, target_demand)
            if not allocation_result:
                self._log_allocation_failure(task, eq_choices, min_required, equipment_analysis)
                return None

            for eq_name, units in allocation_result.items():
                final_allocation[eq_name] = units

        return final_allocation

    def can_allocate(self, task, start, end):
        alloc = self.compute_allocation(task, start, end)
        return alloc is not None

    def release(self, task_id):
        """Release all allocations associated with this task."""
        for eq_name in list(self.allocations.keys()):
            self.allocations[eq_name] = [a for a in self.allocations[eq_name] if a[0] != task_id]

    # ------------------------ Helper Methods ------------------------

    def _normalize_equipment_choices(self, eq_key):
        """Normalize equipment choices to list format."""
        if isinstance(eq_key, (tuple, list)):
            return list(eq_key)
        return [eq_key]

    def _calculate_accelerated_demand(self, min_required, discipline):
        """Calculate accelerated demand with safety limits."""
        acceleration_config = acceleration.get(
            discipline,
            acceleration.get("default", {"factor": 1.0, "max_multiplier": 3.0})
        )
        factor = acceleration_config.get("factor", 1.0)
        max_multiplier = acceleration_config.get("max_multiplier", 3.0)
        accelerated = int(math.ceil(min_required * factor))
        return min(accelerated, int(min_required * max_multiplier))

    def _analyze_equipment_availability(self, eq_choices, start, end, target_demand, task):
        """Analyze equipment availability and constraints for alternatives."""
        equipment_analysis = {}
        total_available = 0

        for eq_name in eq_choices:
            if eq_name not in self.equipment:
                continue

            eq_res = self.equipment[eq_name]
            total_count = int(eq_res.count)
            used_units = self._used_units(eq_name, start, end)
            available_units = max(0, total_count - used_units)
            
            # FIXED: Handle max_equipment as dictionary for task-specific limits
            max_equipment_value = getattr(eq_res, "max_equipment", total_count)
            if isinstance(max_equipment_value, dict):
                # Get task-specific limit
                task_id = getattr(task, 'id', None)
                if task_id and task_id in max_equipment_value:
                    max_per_task = max_equipment_value[task_id]
                else:
                    max_per_task = total_count  # Fallback
            else:
                max_per_task = max_equipment_value
            
            allocatable_units = min(available_units, max_per_task)

            equipment_analysis[eq_name] = {
                'total_count': total_count,
                'used_units': used_units,
                'available_units': available_units,
                'allocatable_units': allocatable_units,
                'max_per_task': max_per_task,
                'hourly_rate': getattr(eq_res, 'hourly_rate', 100),
                'efficiency': getattr(eq_res, 'efficiency', 1.0)
            }

            total_available += allocatable_units

        if total_available < min(1, target_demand):
            return None

        return equipment_analysis

    def _perform_multi_stage_allocation(self, equipment_analysis, min_required, target_demand):
        """Allocate equipment in two stages: minimum and accelerated demand."""
        allocation = {}
        # Stage 1: Ensure minimum
        min_allocation = self._allocate_equipment_set(equipment_analysis, min_required, optimization='min_cost')
        if not min_allocation or sum(min_allocation.values()) < min_required:
            return None

        # Stage 2: Try to meet accelerated demand
        remaining_capacity = self._calculate_remaining_capacity(equipment_analysis, min_allocation)
        additional_demand = target_demand - sum(min_allocation.values())

        if additional_demand > 0 and remaining_capacity > 0:
            additional_allocation = self._allocate_equipment_set(
                equipment_analysis, additional_demand,
                optimization='balanced', existing_allocation=min_allocation
            )
            if additional_allocation:
                for eq_name, units in additional_allocation.items():
                    min_allocation[eq_name] = min_allocation.get(eq_name, 0) + units

        return min_allocation

    def _allocate_equipment_set(self, equipment_analysis, demand, optimization='min_cost', existing_allocation=None):
        allocation = existing_allocation.copy() if existing_allocation else {}
        remaining_demand = demand
        available_eq = self._get_optimized_equipment_list(equipment_analysis, allocation, optimization)

        for eq_name in available_eq:
            if remaining_demand <= 0:
                break

            eq_info = equipment_analysis[eq_name]
            current_alloc = allocation.get(eq_name, 0)
            max_possible = eq_info['allocatable_units'] - current_alloc
            if max_possible <= 0:
                continue

            take = min(max_possible, remaining_demand)
            if take > 0:
                allocation[eq_name] = current_alloc + take
                remaining_demand -= take

        return allocation if remaining_demand == 0 else None

    def _get_optimized_equipment_list(self, equipment_analysis, current_allocation, optimization):
        equipment_list = []
        for eq_name, eq_info in equipment_analysis.items():
            current_alloc = current_allocation.get(eq_name, 0)
            remaining_capacity = eq_info['allocatable_units'] - current_alloc
            if remaining_capacity <= 0:
                continue

            if optimization == 'min_cost':
                score = eq_info['hourly_rate']
            elif optimization == 'max_availability':
                score = -remaining_capacity
            else:  # balanced
                score = eq_info['hourly_rate'] * 0.7 + (-remaining_capacity) * 0.3

            equipment_list.append((eq_name, score))

        equipment_list.sort(key=lambda x: x[1])
        return [eq_name for eq_name, _ in equipment_list]

    def _calculate_remaining_capacity(self, equipment_analysis, current_allocation):
        remaining = 0
        for eq_name, eq_info in equipment_analysis.items():
            current_alloc = current_allocation.get(eq_name, 0)
            remaining += max(0, eq_info['allocatable_units'] - current_alloc)
        return remaining

    def _log_allocation_failure(self, task, eq_choices, min_required, equipment_analysis):
        if equipment_analysis:
            available_str = ", ".join([f"{eq}:{info['allocatable_units']}" 
                                       for eq, info in equipment_analysis.items()])
            print(f"Equipment allocation failed - Task: {task.id}, "
                  f"Required: {min_required}, Available: {available_str}")
        else:
            print(f"Equipment allocation failed - Task: {task.id}, "
                  f"No valid equipment in: {eq_choices}")
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
        q = floor_q.get(task.zone, task.quantity)
        if floor_q is None:  # fallback triggered
            print(f"⚠️ Task {task.base_id} not found in quantity_matrix")
        else:
            qty = floor_q.get(task.zone, task.quantity)
        if qty is None or qty<=1:
            print(f"⚠️ Floor {task.floor} for task {task.base_id} not found in quantity_matrix")
        else:
            print(f"✅ Task {task.base_id}, floor {task.floor} quantity: {qty}")
        task.quantity=q
        return q

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

        if task.task_type == "worker":
            if task.resource_type not in self.workers:
                raise ValueError(f"Worker resource '{task.resource_type}' not found for task {task.id}")
            res = self.workers[task.resource_type]
            # FIXED: Use task-specific productivity rate
            base_prod = self._get_productivity_rate(res, task.base_id, 1.0)
            # worker daily production = base_prod * crews * efficiency
            daily_prod = base_prod * crews * res.efficiency
            if daily_prod <= 0:
                raise ValueError(f"Non-positive worker productivity for {task.id}")
            duration = qty / daily_prod

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
            daily_prod_total = base_prod * total_units * res.efficiency
            
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
            daily_worker_prod = base_prod_worker * crews * worker_res.efficiency
            if daily_worker_prod <= 0:
                raise ValueError(f"Non-positive worker productivity for {task.id}")

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
            duration = max(duration_worker, duration_equip)

        else:
            raise ValueError(f"Unknown task_type: {task.task_type}")

        duration *= task.risk_factor
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
        print(f"for {task.id} duration is {duration_days}")
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
# Topological ordering util for Task objects (for scheduling)
# -----------------------------
def Topo_order_tasks(tasks):
    indegree = {t.id: 0 for t in tasks}
    successors = {t.id: [] for t in tasks}

    for t in tasks:
        for p in t.predecessors:
            indegree[t.id] += 1
            successors[p].append(t.id)

    queue = deque([tid for tid, deg in indegree.items() if deg == 0])
    ordered_ids = []

    while queue:
        current = queue.popleft()
        ordered_ids.append(current)
        for succ in successors[current]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)

    if len(ordered_ids) != len(tasks):
        raise RuntimeError("Cycle detected in task dependencies")

    return ordered_ids
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
def generate_tasks(base_tasks_dict, zone_floors, cross_floor_links=None, ground_disciplines=ground_disciplines):
    """
    Generates multi-floor and multi-zone tasks from base task definitions,
    applying special rules for:
      - Ground tasks → floor 0 only
      - Regular tasks → floors 1..N
      - Special tasks (4.5, 4.6, 4.7) → all floors including 0
        * but on floor 0, only ground or other special predecessors are kept
    Ensures all referenced predecessors actually exist.
    """
    cross_floor_links = cross_floor_links or {}
    ground_disciplines = ground_disciplines or set()
    tasks = []

    # Flatten base tasks for quick lookup (only included)
    base_by_id = {}
    for discipline, base_list in base_tasks_dict.items():
        for base in base_list:
            if getattr(base, "included", True):
                base_by_id[base.id] = base

    # ---------- First pass: generate all task IDs ----------
    task_ids = set()
    for discipline, base_list in base_tasks_dict.items():
        for base in base_list:
            if not getattr(base, "included", True):
                continue

            for zone, max_floor in zone_floors.items():
                # Determine floor range for task generation
                if base.discipline in ground_disciplines:
                    floor_range = [0]
                elif base.id in ["4.5", "4.6", "4.7"]:
                    floor_range = range(max_floor + 1) if getattr(base, "repeat_on_floor", True) else [0]
                else:
                    floor_range = range(1, max_floor + 1) if getattr(base, "repeat_on_floor", True) else [1]

                for f in floor_range:
                    tid = f"{base.id}-F{f}-{zone}"
                    task_ids.add(tid)

    # ---------- Second pass: build tasks with dependencies ----------
    for discipline, base_list in base_tasks_dict.items():
        for base in base_list:
            if not getattr(base, "included", True):
                continue

            is_special = base.id in ["4.5", "4.6", "4.7"]

            for zone, max_floor in zone_floors.items():
                # Determine floor range for task generation
                if base.discipline in ground_disciplines:
                    floor_range = [0]
                elif is_special:
                    floor_range = range(max_floor + 1) if getattr(base, "repeat_on_floor", True) else [0]
                else:
                    floor_range = range(1, max_floor + 1) if getattr(base, "repeat_on_floor", True) else [1]

                for f in floor_range:
                    tid = f"{base.id}-F{f}-{zone}"
                    preds = []

                    # ---------- Special handling: special tasks on floor 0 ----------
                    if is_special and f == 0:
                        for p in base.predecessors:
                            pred_base = base_by_id.get(p)
                            if pred_base and getattr(pred_base, "included", True):
                                # only allow ground or other special predecessors
                                if pred_base.discipline in ground_disciplines or p in ["4.5", "4.6", "4.7"]:
                                    pred_id = f"{p}-F0-{zone}"
                                    if pred_id in task_ids:
                                        preds.append(pred_id)

                    else:
                        # ---------- 1) Regular predecessors ----------
                        for p in base.predecessors:
                            pred_base = base_by_id.get(p)
                            if pred_base and getattr(pred_base, "included", True):
                                # Determine which floor the predecessor should be on
                                if pred_base.discipline in ground_disciplines:
                                    pred_floor = 0
                                else:
                                    pred_floor = f
                                pred_id = f"{p}-F{pred_floor}-{zone}"

                                if pred_id in task_ids:
                                    preds.append(pred_id)
                                else:
                                    print(f"WARNING: Predecessor {pred_id} not found for task {tid}")

                        # ---------- 2) Cross-floor predecessors (previous floor) ----------
                        if f > 0:
                            # Same task from previous floor
                            prev_floor_task = f"{base.id}-F{f-1}-{zone}"
                            if prev_floor_task in task_ids:
                                preds.append(prev_floor_task)

                            # Configured cross-floor links
                            if base.id in cross_floor_links:
                                for p in cross_floor_links[base.id]:
                                    pred_base = base_by_id.get(p)
                                    if pred_base and getattr(pred_base, "included", True):
                                        pred_id = f"{p}-F{f-1}-{zone}"
                                        if pred_id in task_ids:
                                            preds.append(pred_id)

                    # Remove duplicates and self-references
                    preds = [p for p in set(preds) if p != tid]

                    # ---------- Create the Task object ----------
                    tasks.append(Task(
                        id=tid,
                        base_id=base.id,
                        name=base.name,
                        base_duration=base.base_duration,
                        predecessors=preds,
                        discipline=discipline,  # discipline comes from dict key
                        resource_type=base.resource_type,
                        min_crews_needed=base.min_crews_needed,
                        min_equipment_needed=base.min_equipment_needed,
                        task_type=base.task_type,
                        risk_factor=base.risk_factor,
                        zone=zone,
                        floor=f,
                        included=base.included,
                        delay=base.delay
                    ))

    return tasks
def validate_tasks(tasks, workers, equipment, quantity_matrix):

    print("✅ Validation passed: all predecessors exist, no cycles.")
    all_ids = {t.id for t in tasks}
    missing = [(t.id, p) for t in tasks for p in t.predecessors if p not in all_ids]
    if missing:
        # this is serious: some predecessors reference tasks that don't exist in the generated set
        raise ValueError(f"Missing predecessors (referenced but not present): {missing}")

    """
    Validate task data, fill missing quantities/productivities with defaults,
    and print warnings.
    """
    # --- Patch missing quantities ---
    for task in tasks:
        if task.id not in quantity_matrix:
            print(f"⚠️ Warning: No quantity defined for task {task.id} ({task.name}). Defaulting to 1.")
            quantity_matrix[task.id] = {0: {"A": 1}}  # minimal quantity for zone A

    # --- Patch missing productivity for workers ---
    for worker_name, worker in workers.items():
        for task in tasks:
            if task.resource_type == worker.name:
                if task.id not in worker.productivity_rates:
                    print(f"⚠️ Warning: No productivity for worker '{worker_name}' on task {task.id}. Defaulting to 1 unit/hour.")
                    worker.productivity_rates[task.id] = 1

    # --- Patch missing productivity for equipment ---
    for equip_name, equip in equipment.items():
        for task in tasks:
            if task.min_equipment_needed and equip_name in task.min_equipment_needed:
                if task.id not in equip.productivity_rates:
                    print(f"⚠️ Warning: No productivity for equipment '{equip_name}' on task {task.id}. Defaulting to 1 unit/hour.")
                    equip.productivity_rates[task.id] = 1
    try:
        Topo_order_tasks(tasks)
    except ValueError as e:
        raise ValueError(f"Cycle detected: {e}")

    print("✅ Validation passed: all predecessors exist, no cycles.")
    return tasks, workers, equipment, quantity_matrix

    
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
    
    st.header("📅 Generate Project Schedule")
    st.markdown("""
        Upload or define your project inputs:
        - Quantities per task/zone/floor
        - Worker and equipment resources
        - Outputs: Schedule, Gantt, Utilizations
    """)

    # ---------------- Zones & Floors ----------------
    st.subheader("🏗️ Define Zones & Floors")
    zones_floors = {}
    num_zones = st.number_input("Number of zones?", min_value=1, max_value=20, value=2)
    for i in range(num_zones):
        zone_name = st.text_input(f"Zone {i+1} Name", value=f"Zone_{i+1}", key=f"zone_{i}")
        max_floor = st.number_input(f"Max floors for {zone_name}", min_value=0, max_value=100, value=5, key=f"floor_{i}")
        zones_floors[zone_name] = max_floor

    # ---------------- Generate Default Templates ----------------
    st.subheader("📁 Generate Default Excel Templates")
    if st.button("Generate Default Templates"):
        try:
            qty_file = generate_quantity_template(BASE_TASKS, zones_floors)
            with open(qty_file, "rb") as f:
                st.download_button("⬇️ Download Quantity Template", f, file_name="quantity_template.xlsx")

            worker_file = generate_worker_template(workers)
            with open(worker_file, "rb") as f:
                st.download_button("⬇️ Download Worker Template", f, file_name="worker_template.xlsx")

            equip_file = generate_equipment_template(equipment)
            with open(equip_file, "rb") as f:
                st.download_button("⬇️ Download Equipment Template", f, file_name="equipment_template.xlsx")
        except Exception as e:
            st.error(f"Failed to generate default templates: {e}")

    # ---------------- Upload User Files ----------------
    st.subheader("📤 Upload Your Excel Files")
    quantity_file = st.file_uploader("Quantity Matrix (Excel)", type=["xlsx"])
    worker_file = st.file_uploader("Worker Template (Excel)", type=["xlsx"])
    equipment_file = st.file_uploader("Equipment Template (Excel)", type=["xlsx"])
    start_date = st.date_input("Project Start Date", value=pd.Timestamp.today())

    # ---------------- Generate Schedule ----------------
    if st.button("🚀 Generate Schedule"):
        if not quantity_file:
            st.warning("Please upload the Quantity Matrix Excel.")
            return
        if not worker_file:
            st.warning("Please upload the Worker Template Excel.")
            return
        if not equipment_file:
            st.warning("Please upload the equipment Template Excel.")
            return
        # Parse uploaded files
        try:
            df_qty = pd.read_excel(quantity_file)
            quantity_matrix = parse_quantity_excel(df_qty)
        except Exception as e:
            st.error(f"Error parsing quantity matrix: {e}")
            return

        # Worker data
        if worker_file:
            try:
                df_worker = pd.read_excel(worker_file)
                workers_used = parse_worker_excel(df_worker)
            except Exception as e:
                st.error(f"Error parsing worker template: {e}")
                return

        # Equipment data
        if equipment_file:
            try:
                df_equip = pd.read_excel(equipment_file)
                equipment_used = parse_equipment_excel(df_equip)
            except Exception as e:
                st.error(f"Error parsing equipment template: {e}")
                return

        # Run scheduling logic
        with st.spinner("Generating schedule..."):
            try:
                schedule, output_folder = run_schedule(
                    zones_floors,
                    quantity_matrix,
                    start_date,
                    workers_dict=workers_used,
                    equipment_dict=equipment_used
                )
            except Exception as e:
                st.error(f"Failed to generate schedule: {e}")
                return

        st.success("✅ Schedule generated successfully!")

        # ---------------- Download all generated files ----------------
        st.subheader("📂 Download Generated Files")
        for file_name in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file_name)
            with open(file_path, "rb") as f:
                st.download_button(f"⬇️ {file_name}", f, file_name=file_name)

        # ---------------- Download Interactive Gantt HTML ----------------
        st.subheader("📊 Interactive Gantt Chart")
        
        try:
            from reporting import generate_interactive_gantt
            gantt_file = os.path.join(output_folder, f"interactive_gantt_{start_date.strftime('%Y%m%d')}.html")
            generate_interactive_gantt(schedule, gantt_file)
            if os.path.exists(gantt_file):
                with open(gantt_file, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download Interactive Gantt Chart (HTML, start {start_date.strftime('%Y-%m-%d')})",
                        data=f,
                        file_name=f"interactive_gantt_{start_date.strftime('%Y%m%d')}.html",
                        mime="text/html"
                    )
        except Exception as e:
            st.warning(f"Interactive Gantt could not be generated: {e}")

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
        
