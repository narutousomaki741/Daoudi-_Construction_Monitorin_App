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
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict, deque
import logging
import loguru
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import save_excel_file



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
ground_disciplines=["Préliminaire","Terrassement","Fondations"]
# -----------------------------
# Data classes: workers & equip
# -----------------------------
@dataclass
class WorkerResource:
    name: str
    count: int
    hourly_rate: float
    productivity_rates: dict   # {base_task_id: units_per_workday}
    skills: List[str]
    max_crews: Optional[int] = None
    overtime_factor: float = 1.5
    efficiency: float = 1.0


@dataclass
class EquipmentResource:
    name: str
    count: int
    hourly_rate: float
    productivity_rates: dict   # {base_task_id: units_per_workday}
    max_equipment: Optional[int] = None
    type: str = "general"
    efficiency: float = 1.0


@dataclass
class BaseTask:
    id: str
    name: str
    discipline: str
    resource_type: str                   # logical resource name (worker pool) for worker/hybrid tasks      
    predecessors: List[str] = field(default_factory=list)
    task_type: str = "worker"# worker | equipment | hybrid
    min_crews_needed: Optional[int] = None
    min_equipment_needed: Optional[Dict[str, int]] = None
    base_duration: int = None
    risk_factor: float = 1.0
    repeat_on_floor: bool=True
    included: bool = True
    delay: int = 0

@dataclass
class Task:
    id: str
    base_id: str
    name: str
    base_duration: int
    predecessors: list
    discipline: str
    resource_type: str
    min_crews_needed: Optional[int] = None
    min_equipment_needed: Optional[Dict[str, int]] = None
    allocated_crews: int=None
    allocated_equipments:Optional[Dict[str, int]] = None
    task_type: str = "worker"
    quantity: float = 250.0
    risk_factor: float = 1.0
    weather_sensitive: bool = False
    floor: int = 0
    zone: str = ""
    constraints: list = None
    included: bool = True
    earliest_start: Optional[datetime] = None
    earliest_finish: Optional[datetime] = None
    latest_start: Optional[datetime] = None
    latest_finish: Optional[datetime] = None
    delay: int = 0


    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        # default min to the requested crews if not provided
        if self.min_crews_needed is None:
            # ensure at least 1 unless crews_needed is 0
            self.min_crews_needed = max(1, int(self.min_crews_needed)) if getattr(self, "min_crews_needed", 0) else 0
        # default min equipment if not provided: use equipment_needed itself
        if self.min_equipment_needed is None:
            self.min_equipment_needed = {}
import pandas as pd
from io import BytesIO

def load_quantity_excel(file):
    """Override default quantities from uploaded Excel."""
    if file is None:
        return None
    df = pd.read_excel(BytesIO(file.read()))
    updated_dict = {}
    for _, row in df.iterrows():
        zone = row.get("Zone")
        floor = row.get("Floor")
        task_id = row.get("TaskID")
        qty = row.get("Quantity")
        if pd.notna(zone) and pd.notna(floor) and pd.notna(task_id) and pd.notna(qty):
            updated_dict[(task_id, zone, floor)] = qty
    return updated_dict

def load_worker_excel(file):
    """Override default worker counts/productivity from Excel."""
    if file is None:
        return None
    df = pd.read_excel(BytesIO(file.read()))
    updated_workers = {}
    for _, row in df.iterrows():
        name = row.get("Worker")
        count = row.get("Count")
        productivity = row.get("Productivity")
        if pd.notna(name):
            if name in workers:  # keep the existing object and just override
                obj = workers[name]
                if pd.notna(count):
                    obj.count = count
                if pd.notna(productivity):
                    for task, val in productivity.items():
                        obj.productivity_rates[task] = val
                updated_workers[name] = obj
    return updated_workers

def load_equipment_excel(file):
    """Override default equipment counts/productivity from Excel."""
    if file is None:
        return None
    df = pd.read_excel(BytesIO(file.read()))
    updated_equipment = {}
    for _, row in df.iterrows():
        name = row.get("Equipment")
        count = row.get("Count")
        productivity = row.get("Productivity")
        if pd.notna(name):
            if name in equipment:
                obj = equipment[name]
                if pd.notna(count):
                    obj.count = count
                if pd.notna(productivity):
                    for task, val in productivity.items():
                        obj.productivity_rates[task] = val
                updated_equipment[name] = obj
    return updated_equipment

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

        # requested minimum and maximum for the task
        
        min_needed = max(1, int(task.min_crews_needed))

        # acceleration config may increase desired crews (factor) but we cap by task max and pool limits
        acc = acceleration.get(
        task.discipline,
        acceleration.get("default", {"factor": 1.0})
        )
        factor = acc.get("factor", 1.0  )

        # ideal after acceleration (but must be <= disc_max and <= per-res max)
        candidate = int(math.ceil(min_needed * factor))
        # clamp to resource's own per-task crew limit if present and meaningful
        res_max = getattr(res, "max_crews", None)
        if res_max is not None and res_max > 0:
               candidate = min(candidate, int(res_max))

        # final allocation is the maximum we can give within [min_needed, candidate] limited by available
        allocated = min(candidate, available)

        # If allocated is less than minimum, fail
        if allocated < min_needed:
            # debug: show why
            print(f"[ALLOC FAIL] {task.id} pool={total_pool} used={used} available={available} min_needed={min_needed} candidate={candidate}")
            return 0

        # Otherwise return allocated (could be >= min_needed and <= candidate)
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
            equipment_analysis = self._analyze_equipment_availability(eq_choices, start, end, target_demand)
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

    def _analyze_equipment_availability(self, eq_choices, start, end, target_demand):
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
            max_per_task = getattr(eq_res, "max_equipment", total_count)
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
            logger.warning(f"Equipment allocation failed - Task: {task.id}, "
                          f"Required: {min_required}, Available: {available_str}")
        else:
            logger.warning(f"Equipment allocation failed - Task: {task.id}, "
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
            base_prod = res.productivity_rates.get(task.base_id, 1.0)
            # worker daily production = base_prod * crews * efficiency
            daily_prod = base_prod * crews * res.efficiency
            if daily_prod <= 0:
                raise ValueError(f"Non-positive worker productivity for {task.id}")
            duration = qty / daily_prod

        elif task.task_type == "equipment":
            if not eq_alloc:
                raise ValueError(f"Equipment task {task.id} has no equipment specified")
            normalized_eq_alloc = {}
            for eq_key, units in eq_alloc.items():
                if isinstance(eq_key, (tuple, list)):
            # pick the first matching equipment in self.equipment
                    chosen = next((e for e in eq_key if e in self.equipment), None)
                    if chosen is None:
                        raise ValueError(f"Equipment {eq_key} not found for task {task.id}")
                    normalized_eq_alloc[chosen] = normalized_eq_alloc.get(chosen, 0) + units
                else:
                    normalized_eq_alloc[eq_key] = normalized_eq_alloc.get(eq_key, 0) + units

            eq_alloc = normalized_eq_alloc


            daily_prod_total = 0.0
            for eq_name, units in eq_alloc.items():
                if eq_name not in self.equipment:
                    raise ValueError(f"Equipment '{eq_name}' not found for task {task.id}")
                res = self.equipment[eq_name]
                base_prod = res.productivity_rates.get(task.base_id, 1.0)
                daily_prod_total += base_prod * units * res.efficiency
            if daily_prod_total <= 0:
                raise ValueError(f"Non-positive total equipment productivity for {task.id}")
            duration = qty / daily_prod_total

        elif task.task_type == "hybrid":
        # worker-limited
            if task.resource_type not in self.workers:
                raise ValueError(f"Worker resource '{task.resource_type}' not found for task {task.id}")
            worker_res = self.workers[task.resource_type]
            base_prod_worker = worker_res.productivity_rates.get(task.base_id, 1.0)
            daily_worker_prod = base_prod_worker * crews * worker_res.efficiency
            if daily_worker_prod <= 0:
                raise ValueError(f"Non-positive worker productivity for {task.id}")

        # equipment-limited: compute effective daily production per equipment bottleneck:
          #   durations_equip = []
          #   if eq_alloc:
          #       for eq_name, units in eq_alloc.items():
          #           if eq_name not in self.equipment:
          #               raise ValueError(f"Equipment '{eq_name}' not found for task {task.id}")
          #           eq_res = self.equipment[eq_name]
          #           base_prod_eq = eq_res.productivity_rates.get(task.base_id, 1.0)
          #           daily_eq_prod = base_prod_eq * units * eq_res.efficiency
          #           if daily_eq_prod <= 0:
          #               durations_equip.append(float("inf"))
           #          else:
           #              durations_equip.append(qty / daily_eq_prod)
           #      duration_equip = max(durations_equip) if durations_equip else float("inf")
           
           
           
           #  else:
            #     duration_equip = float("inf")

        # worker duration
            duration_worker = qty / daily_worker_prod

        # realistic is the max of worker-limited and equipment-limited
           #  duration = max(duration_worker, duration_equip)
            duration = duration_worker
        else:
            raise ValueError(f"Unknown task_type: {task.task_type}")

        duration *= task.risk_factor
        shift_factor = SHIFT_CONFIG.get(task.discipline, SHIFT_CONFIG.get("default", 1.0))
        duration = duration /shift_factor
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
        
         # -------------------------
        # Precompute durations (fail early)
        # -------------------------
        for tid, t in self.task_map.items():
            try:
                # compute a nominal duration using nominal resources (no allocations)
                d = self.duration_calc.calculate_duration(t)
                if not isinstance(d, int) or d < 0:
                    raise ValueError(f"Computed invalid duration {d!r}")
            except Exception as e:
                print(f"[DUR ERROR] Task {tid}: cannot compute nominal duration before scheduling => {e!r}")
                # Re-raise so you see the task and stop early — avoids infinite loops later
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

            end_date = self.calendar.add_workdays(start_date, duration_days)

            forward_attempts = 0
            max_forward = 3000
            allocated_crews = None
            allocated_equipments = None
            self.worker_manager.release(task.id)
            self.equipment_manager.release(task.id)
            if duration_days==0:
                start_date = self._earliest_start_from_preds(task, schedule)
                end_date = start_date
                allocated_crews = 0
                allocated_equipments = {}
                schedule[task.id] = (start_date, end_date)
                unscheduled.remove(task.id)
                task.allocated_crews = allocated_crews
                task.allocated_equipments = allocated_equipments
    # update successors
                for succ in [s for s in self.task_map if task.id in self.task_map[s].predecessors]:
                    pred_count[succ] -= 1
                    if pred_count[succ] == 0:
                        ready.append(succ)
                        continue # instantaneous task
            else:
                
                while True:
                # === 1. Compute allocations on this window ===
                    possible_crews = None
                    if task.task_type in ("worker", "hybrid"):
                        possible_crews = self.worker_manager.compute_allocation(task, start_date, end_date)

                    possible_equip = {}
                    if task.task_type in ("equipment", "hybrid") and (task.min_equipment_needed or {}):
                       possible_equip = self.equipment_manager.compute_allocation(task, start_date, end_date) or {}
  
                  # === 2. Check feasibility vs min requirements ===
                    min_crews = getattr(task, "min_crews_needed", max(1, task.min_crews_needed))
                    feasible_workers = (possible_crews is not None and possible_crews >= min_crews) if task.task_type in ("worker", "hybrid") else True
                    feasible_equip = True
                    if task.task_type in ("equipment", "hybrid") and (task.min_equipment_needed or {}):
                        for eq_key, min_req in task.min_equipment_needed.items():
                            eq_choices = eq_key if isinstance(eq_key, (tuple, list)) else (eq_key,)
                            allocated_total = sum(possible_equip.get(eq, 0) for eq in eq_choices)
                            if allocated_total < min_req:
                                feasible_equip = False
                                break

                     # === 3. If feasible, calculate duration using ACTUAL allocation ===
                    if ((task.task_type == "worker" and feasible_workers) or
                        (task.task_type == "equipment" and feasible_equip) or
                        (task.task_type == "hybrid" and feasible_workers and feasible_equip)):

                        dur_try = self.duration_calc.calculate_duration(
                            task,
                           allocated_crews=possible_crews if possible_crews else None,
                           allocated_equipments=possible_equip if possible_equip else None
                         )
                        end_try = self.calendar.add_workdays(start_date, dur_try)

                        # === 4. Re-check allocations on the final window ===
                        final_crews = None
                        if task.task_type in ("worker", "hybrid"):
                            final_crews = self.worker_manager.compute_allocation(task, start_date, end_try)

                        final_equip = {}
                        if task.task_type in ("equipment", "hybrid") and (task.min_equipment_needed or {}):
                            final_equip = self.equipment_manager.compute_allocation(task, start_date, end_try) or {}

                        feasible_final_workers = True
                        if task.task_type in ("worker", "hybrid"):
                            feasible_final_workers = (final_crews is not None and final_crews >= min_crews)

                        feasible_final_equip = True
                        if task.task_type in ("equipment", "hybrid"):
                            for eq_key, min_req in task.min_equipment_needed.items():
                                eq_choices = eq_key if isinstance(eq_key, (tuple, list)) else (eq_key,)
                                allocated_total = sum(final_equip.get(eq, 0) for eq in eq_choices)
                                if allocated_total < min_req:
                                    feasible_final_equip = False
                                    break
                        if feasible_final_workers and feasible_final_equip:
                             # === 5. Commit allocation ===
                            allocated_crews = final_crews if final_crews else None
                            allocated_equipments = final_equip if final_equip else {}

                            if allocated_crews:
                                self.worker_manager.allocate(task, start_date, end_try,allocated_crews)
                            if allocated_equipments:
                                self.equipment_manager.allocate(task, start_date, end_try,allocated_equipments)

                            duration_days = dur_try
                            end_date = end_try
                            break  # scheduled successfully

                     # === 6. If not feasible, shift window ===
                    if not feasible_workers and task.task_type in ("worker", "hybrid"):
                        print(f"   ❌ {task.id}: insufficient crews at {start_date}")
                    if not feasible_equip and task.task_type in ("equipment", "hybrid"):
                        print(f"   ❌ {task.id}: insufficient equipment at {start_date}")
                    
                    start_date = self.calendar.add_workdays(start_date, 1)
                    end_date = self.calendar.add_workdays(start_date, duration_days)
                    forward_attempts += 1
                    if forward_attempts > max_forward:
                        print(f"[ATTEMPT {forward_attempts}] task={task.id} start={start_date} end={end_date}")
                        print(f"  possible_crews={possible_crews} feasible_workers={feasible_workers}")
                        print(f"  possible_equip={possible_equip} feasible_equip={feasible_equip}")
                        print(f"[DEBUG LOOP] Task {task.id} start={start_date}")
                        print(f"  dur_try={dur_try}")
                        print(f"  possible_crews={possible_crews}")
                        print(f"  final_crews={final_crews}")
                        print(f"  feasible_final_workers={feasible_final_workers}")
                        print(f"  feasible_final_equip={feasible_final_equip}")
                        raise RuntimeError(f"Could not find resource window for task {task.id} after {max_forward} attempts.")
            # === 7. Final dependency enforcement ===
            for p in task.predecessors:
                pred_end = schedule[p][1]
                if start_date < pred_end:
                    raise RuntimeError(
                        f"Dependency violation: Task {task.id} starts {start_date} before predecessor {p} ends {pred_end}"
                    )

            # === 8. Record schedule ===
            schedule[task.id] = (start_date, end_date,)
            if task.id in unscheduled:
                unscheduled.remove(task.id)
            task.allocated_crews=allocated_crews
            task.allocated_equipments=allocated_equipments
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

    # check for cycles (Topo_order_tasks will raise)

  
# -----------------------------
# Example resources & tasks (you can replace these with your data)
# -----------------------------
workers = {
    "BétonArmée": WorkerResource(
        "BétonArmée", count=200, hourly_rate=18,
        productivity_rates={"3.1":5,"3.2":5,"2.1":5,"3.4": 5, "3.5":5,"4.1": 12,"3.8":5,"3.9":5,
                        "3.7":5,"3.10": 10,"3.11": 10,"3.13": 10,"4.1": 10,"4.3": 10,"4.4": 10
                       ,"4.6": 10,"4.7": 10 ,"4.9": 10,"4.10": 10},
        skills=["BétonArmée"],max_crews=25
    ),
    "Férrailleur": WorkerResource(
        "Férrailleur", count=85, hourly_rate=18,
        productivity_rates={"3.3": 400, "3.6": 180,"3.12": 300,"4.2": 180,"4.5": 300,
                            "4.8": 120},
        skills=["BétonArmée"],max_crews=25
    ),
    "Topograph": WorkerResource(
        "Topograph", count=5, hourly_rate=18,
        productivity_rates={"1.3": 100},
        skills=["BétonArmée"],max_crews=10
    ),
    "ConstMétallique": WorkerResource(
        "ConstMétallique", count=3, hourly_rate=60,
        productivity_rates={"9.2": 8},
        skills=["ConstMétallique"],max_crews=10
    ),
    "Maçonnerie": WorkerResource(
        "Maçonnerie", count=84, hourly_rate=40,
        productivity_rates={"5.1": 10},
        skills=["Maçonnerie"],max_crews=25
    ),
     "Cloisennement": WorkerResource(
        "Cloisennement", count=84, hourly_rate=40,
        productivity_rates={"5.1": 10},
        skills=["Cloisennement"],max_crews=25
    ),
    "Etanchiété": WorkerResource(
        "Etanchiété", count=83, hourly_rate=40,
        productivity_rates={"5.2": 10},
        skills=["Etanchiété"],max_crews=25
    ),
    "Revetement": WorkerResource(
        "Revetement", count=84, hourly_rate=40,
        productivity_rates={"5.3": 15, "5.4": 10},
        skills=["Carrelage", "Marbre"],max_crews=15
    ),
    "Peinture": WorkerResource(
        "Peinture", count=8, hourly_rate=40,
        productivity_rates={"5.4": 10, "5.5": 25},
        skills=["Peinture"],max_crews=15
    ),
     "Enduit": WorkerResource(
        "Enduit", count=8, hourly_rate=40,
        productivity_rates={"5.4": 10, "5.5": 25},
        skills=["Enduit"],max_crews=15
    ),
}

equipment = {
    "Chargeuse": EquipmentResource(
        "Chargeuse", count=160, hourly_rate=100,
        productivity_rates={ "2.2": 120, "2.3": 20,"2.4": 40,"2.5": 20,"2.6": 20,
                            "2.7": 20,"2.9": 20},
        type="Terrassement",max_equipment=6
    ),
    "Bulldozer": EquipmentResource(
        "Bulldozer", count=16, hourly_rate=100,
        productivity_rates={"2.1": 15, "2.2": 200, "2.3": 20},
        type="Terrassement",max_equipment=6
    ),
    "Pelle": EquipmentResource(
        "Pelle", count=16, hourly_rate=100,
        productivity_rates={"2.1": 15, "2.2": 200, "2.3": 20},
        type="Terrassement",max_equipment=6
    ),
    "Tractopelle": EquipmentResource(
        "Tractopelle", count=16, hourly_rate=100,
        productivity_rates={"2.1": 15, "2.2": 200, "2.3": 20},
        type="Terrassement",max_equipment=6
    ),
    "Niveleuse": EquipmentResource(
        "Niveleuse", count=16, hourly_rate=100,
        productivity_rates={"2.1": 15, "2.2": 200, "2.3": 20},
        type="Terrassement",max_equipment=6
    ),
    "Compacteur": EquipmentResource(
        "Compacteur", count=16, hourly_rate=100,
        productivity_rates={ "2.9": 20},
        type="Terrassement",max_equipment=6
    ),
    "Grue à tour": EquipmentResource(
        "Crane", count=80, hourly_rate=150,
        productivity_rates={"5.1": 10},
        type="Levage",max_equipment=8
    ),
    "Grue mobile": EquipmentResource(
        "Crane", count=90, hourly_rate=150,
        productivity_rates={"5.1": 10},
        type="Levage",max_equipment=8
    ),
     "Nacelle": EquipmentResource(
        "Nacelle", count=16, hourly_rate=100,
        productivity_rates={"2.1": 15, "2.2": 200, "2.3": 20},
        type="Levage",max_equipment=6
    ),
    "Pump": EquipmentResource(
        "Pump", count=30, hourly_rate=190,
        productivity_rates={"3.5": 14, "4.1": 16},
        type="Bétonnage",max_equipment=3
    ),
    "Camion": EquipmentResource(
        "Camion", count=9, hourly_rate=190,
        productivity_rates={"2.10": 120, "2.8": 120},
        type="Transport",max_equipment=3
    ),
    "Bétonier": EquipmentResource(
        "Bétonier", count=9, hourly_rate=190,
        productivity_rates={"3.5": 14, "4.1": 16},
        type="Bétonnage",max_equipment=3
    ),
    "Manito": EquipmentResource(
        "Manito", count=19, hourly_rate=190,
        productivity_rates={"3.5": 14, "4.1": 16},
        type="Transport",max_equipment=8
    ),
}

BASE_TASKS = {
    "Préliminaire": [
        BaseTask(
            id="1.1", name="Validation du Plan_implantation_EXE", discipline="Préliminaire",
            resource_type="BétonArmée", task_type="hybrid",base_duration=0,
            predecessors=[], repeat_on_floor=False
        ),
        BaseTask(
            id="1.2", name="Bases vie", discipline="Préliminaire",
            resource_type="BétonArmée", task_type="worker",
            repeat_on_floor=False,min_crews_needed=2,predecessors=["1.1"]
        ),
        BaseTask(
            id="1.3", name="Levée Topographique", discipline="Préliminaire",
            resource_type="Topograph", predecessors=["1.1"],base_duration=2,
            repeat_on_floor=False,min_crews_needed=2
        ),
         BaseTask(
            id="1.4", name="Installations temporaires", discipline="Préliminaire",
            resource_type="BétonArmée", predecessors=["1.2"],base_duration=4,
            repeat_on_floor=False,min_crews_needed=2
        ),
         BaseTask(
            id="1.5", name="Signalisation", discipline="Préliminaire",
            resource_type="BétonArmée", predecessors=["1.1"],base_duration=2,
            repeat_on_floor=False,min_crews_needed=2
        ),
    ],
    "Terrassement": [
        BaseTask(
            id="2.1", name="Validation des PLAN_NIVEAUX_EXE", discipline="Terrassement",resource_type="BétonArmée", task_type="equipment",
             min_crews_needed=2, predecessors=["1.3","1.1"], repeat_on_floor=False,base_duration=0
        ),
        BaseTask(
            id="2.2", name="Décapage & nettoyage", discipline="Terrassement",resource_type="BétonArmée", task_type="equipment",
            min_equipment_needed={"Chargeuse": 1,"Bulldozer":1}, min_crews_needed=2, predecessors=["2.1","1.2"], repeat_on_floor=False
        ),
        BaseTask(
            id="2.3", name="Déviation et protection réseaux existants", discipline="Terrassement", resource_type="BétonArmée", task_type="equipment",
            min_equipment_needed={"Pelle": 1,"Chargeuse":1}, min_crews_needed=3,predecessors=["2.2"], repeat_on_floor=False
        ),
        BaseTask(
            id="2.4", name="Excavation en masse", discipline="Terrassement", resource_type="BétonArmée", 
            task_type="equipment", min_equipment_needed={"Chargeuse": 1,("Pelle","Tractopelle"):1},
             min_crews_needed=3,predecessors=["2.2","2.3"], repeat_on_floor=False
        ),
        BaseTask(
            id="2.5", name="Souténement temporaire", discipline="Terrassement", resource_type="BétonArmée",
             task_type="equipment", min_equipment_needed={("Pelle","Tractopelle"): 1,"Chargeuse":1}, min_crews_needed=3,
            predecessors=["2.4"], repeat_on_floor=False,included=False
        ),
        BaseTask(
            id="2.6", name="Excavation des tranchées de fondations", discipline="Terrassement", resource_type="BétonArmée",
              task_type="equipment",min_equipment_needed={"Manito": 1,("Pelle","Tractopelle"):1}, min_crews_needed=3,
            predecessors=["2.5","2.4"], repeat_on_floor=False
        ),
        BaseTask(
            id="2.7", name="Stabilisation et protection des talus", discipline="Terrassement", resource_type="BétonArmée", 
            task_type="equipment",min_equipment_needed={"Chargeuse": 1}, min_crews_needed=3,
            predecessors=["2.4"], repeat_on_floor=False,included=False
        ),
        BaseTask(
            id="2.8", name="Aport du matériaux de remblais", discipline="Terrassement", resource_type="BétonArmée", task_type="equipment",
            min_equipment_needed={"Chargeuse": 1,"Camion":1}, min_crews_needed=3,
            predecessors=["2.6"], repeat_on_floor=False,
        ),
        BaseTask(
            id="2.9", name="Remblais+Compactage", discipline="Terrassement",resource_type="BétonArmée", 
            task_type="equipment",min_equipment_needed={"Chargeuse": 1, "Compacteur": 1},
              min_crews_needed=3,predecessors=["2.8","2.7","2.6"], repeat_on_floor=False ),
        BaseTask(
            id="2.10", name="Export du matériaux de déblais", discipline="Terrassement", resource_type="BétonArmée", task_type="equipment",
            min_equipment_needed={"Chargeuse": 1,"Camion":1}, min_crews_needed=3,predecessors=["2.6"], repeat_on_floor=False
        ),
    ],
    "Fondations": [
        BaseTask(
            id="3.1", name="Validation du Plans_couffrage/ferraillage_Fondations_EXE", discipline="Fondations",resource_type="BétonArmée",
            task_type="hybrid",base_duration=0,predecessors=["2.1"], repeat_on_floor=False),
         BaseTask(
            id="3.2", name="Préparation de la couche de forme", discipline="Fondations",
            resource_type="BétonArmée", task_type="hybrid",base_duration=1,
            min_equipment_needed={"Bétonier": 1}, min_crews_needed=2,
            predecessors=["3.1","2.6"], repeat_on_floor=False ),
         BaseTask(
            id="3.3", name="Préparation du ferraillage des semelles", discipline="Fondations",
            resource_type="Férrailleur", task_type="worker", min_crews_needed=2,
            predecessors=["3.1","2.10","2.9"], repeat_on_floor=False ),
        BaseTask(
            id="3.4", name="Coffrage et Pose du armatures des semelles", discipline="Fondations",
            resource_type="BétonArmée", task_type="hybrid",
            min_equipment_needed={"Grue à tour": 1}, min_crews_needed=2,
            predecessors=["3.3","3.2"], repeat_on_floor=False),
         BaseTask(
            id="3.5", name="Bétonnage des semelles", discipline="Fondations",
            resource_type="BétonArmée", task_type="hybrid", delay=5,
            min_equipment_needed={"Pump": 1,"Bétonier":1}, min_crews_needed=2,
            predecessors=["3.4"], repeat_on_floor=False),
         BaseTask(
            id="3.6", name="Préparation du armatures des murs de fondations", discipline="Fondations",
            resource_type="Férrailleur", task_type="worker", min_crews_needed=2,
            predecessors=["3.1"], repeat_on_floor=False ),
         BaseTask(
            id="3.7", name="Coffrage et pose du armatures des murs de fondations", discipline="Fondations",
            resource_type="BétonArmée", task_type="hybrid",
            min_equipment_needed={"Grue à tour": 1}, min_crews_needed=2,
            predecessors=["3.6","3.5"], repeat_on_floor=False),
         BaseTask(
            id="3.8", name="Bétonnage des murs de fondations", discipline="Fondations",
            resource_type="BétonArmée", task_type="hybrid",base_duration=2,
            min_equipment_needed={"Pump": 1,"Bétonier":1}, min_crews_needed=2,
            predecessors=["3.7"], repeat_on_floor=False),
         BaseTask(
            id="3.9", name="Installation du système de drainage", discipline="Fondations",
            resource_type="BétonArmée", task_type="hybrid",
            min_equipment_needed={("Chargeuse","Pelle"):1,"Manito":1}, min_crews_needed=2,
            predecessors=["3.8"], repeat_on_floor=False),
        BaseTask(
            id="3.10", name="Etanchiété des fondations", discipline="Fondations",
            resource_type="BétonArmée", task_type="worker",
            min_equipment_needed={"Manito": 1}, min_crews_needed=2,
            predecessors=["3.9"], repeat_on_floor=False),
        BaseTask(
            id="3.11", name="Réseau sous dallage", discipline="Fondations",
            resource_type="BétonArmée", task_type="hybrid",
            min_equipment_needed={"Manito": 1,"Pelle":1}, min_crews_needed=2,
            predecessors=["3.10"], repeat_on_floor=False ),
        BaseTask(
            id="3.12", name="Pose du armatures du dallage", discipline="Fondations",
            resource_type="Férrailleur", task_type="hybrid",
            min_equipment_needed={"Manito": 1}, min_crews_needed=2,
            predecessors=["3.11"], repeat_on_floor=False ),
        BaseTask(
            id="3.13", name="Bétonnage du dallage", discipline="Fondations",
            resource_type="BétonArmée", task_type="hybrid",base_duration=2,
            min_equipment_needed={"Pump": 1,"Bétonier":1}, min_crews_needed=2,
            predecessors=["3.12"], repeat_on_floor=False  ),
        
    ],
    "Superstructure": [
        BaseTask(
            id="4.1", name="Validation du Plans_couffrage/ferraillage_EXE", discipline="Superstructure",base_duration=0,
            resource_type="BétonArmée", task_type="hybrid",min_equipment_needed={"Grue à tour": 1}, min_crews_needed=2, predecessors=["3.1"] ),
        BaseTask(
            id="4.2", name="Préparations des armatures des poteaux/voiles", discipline="Superstructure",
            resource_type="Férrailleur", task_type="hybrid",
            min_equipment_needed={"Grue mobile": 1, "Pump": 1}, min_crews_needed=2,
            predecessors=["4.1"]),
        
         BaseTask(
            id="4.3", name="Coffrage+pose des armatures des poteaux/voiles", discipline="Superstructure",
            resource_type="BétonArmée", task_type="hybrid",
            min_equipment_needed={"Grue mobile": 1, "Pump": 1}, min_crews_needed=2,
            predecessors=["4.1", "4.2"]),
        BaseTask(
            id="4.4", name="Bétonnage des poteaux/voiles", discipline="Superstructure",
            resource_type="BétonArmée", task_type="hybrid",base_duration=2,
            min_equipment_needed={"Grue mobile": 1, "Pump": 1}, min_crews_needed=2,
            predecessors=["4.3"]),
        BaseTask(
            id="4.5", name="Préparation du armatures des poutres/plancher-Haut", discipline="Superstructure",
            resource_type="Férrailleur", task_type="hybrid",
            min_equipment_needed={"Grue à tour": 1}, min_crews_needed=2,
            predecessors=["4.1","4.4"] ),
        BaseTask(
            id="4.6", name="Coffrage+pose des armatures des poutres/plancher-Haut", discipline="Superstructure",
            resource_type="BétonArmée", task_type="hybrid",
            min_equipment_needed={"Grue mobile": 1, "Pump":1}, min_crews_needed=2,
            predecessors=["4.5"] ),
        BaseTask(
            id="4.7", name="Bétonnages des poutres/planchier-Haut", discipline="Superstructure",
            resource_type="BétonArmée", task_type="hybrid",base_duration=2,
            min_equipment_needed={"Grue mobile": 1, "Pump": 1}, min_crews_needed=2,
            predecessors=["4.6"] ),
        
        BaseTask(
            id="4.8", name="Préparations des armatures des escaliers", discipline="Superstructure",
            resource_type="Férrailleur", task_type="hybrid",
            min_equipment_needed={"Grue à tour": 1, "Pump": 1}, min_crews_needed=2,
            predecessors=["4.1"]),
        BaseTask(
            id="4.9", name="Coffrage+pose des armatures des escaliers", discipline="Superstructure",
            resource_type="BétonArmée", task_type="hybrid",
            min_equipment_needed={"Grue à tour": 1, "Pump":1}, min_crews_needed=2,
            predecessors=["4.8"]),
        BaseTask(
            id="4.10", name="Bétonnage des escaliers", discipline="Superstructure",
            resource_type="BétonArmée", task_type="hybrid",
            min_equipment_needed={"Grue à tour": 1, "Pump": 1}, min_crews_needed=2,
            predecessors=["4.9"]),
        ],
        
    "SecondeOeuvre": [
        BaseTask(
            id="5.1", name="Maçonnerie", discipline="SecondeOeuvre",
            resource_type="Maçonnerie", task_type="worker",
            min_equipment_needed={"Grue à tour": 1}, min_crews_needed=2,
            predecessors=[]
        ),
         BaseTask(
            id="5.2", name="Cloisennement", discipline="SecondeOeuvre",
            resource_type="Cloisennement", task_type="hybrid",
            min_equipment_needed={"Grue à tour": 1}, min_crews_needed=2,
            predecessors=[]
        ),
        BaseTask(
            id="5.3", name="Etanchiété", discipline="SecondeOeuvre",
            resource_type="Etanchiété", task_type="worker",
            min_equipment_needed={"Grue à tour": 1}, min_crews_needed=2,
            predecessors=["5.1","5.2"]
        ),
        BaseTask(
            id="5.4", name="Carrelage", discipline="SecondeOeuvre",
            resource_type="Revetement", task_type="hybrid",
            min_equipment_needed={"Grue à tour": 1}, min_crews_needed=2,
            predecessors=["5.3"]
        ),
        BaseTask(
            id="5.5", name="Marbre", discipline="SecondeOeuvre",
            resource_type="Revetement", task_type="hybrid",
            min_equipment_needed={"Grue à tour": 1}, min_crews_needed=2,
            predecessors=["5.3"]
        ),
        BaseTask(
            id="5.6", name="Peinture", discipline="SecondeOeuvre",
            resource_type="Peinture", task_type="worker",
            min_crews_needed=2, predecessors=["5.3"]
        ),
        BaseTask(
            id="5.7", name="Enduit", discipline="SecondeOeuvre",
            resource_type="Enduit", task_type="worker",
            min_crews_needed=2, predecessors=["5.6"]
        ),
    ],
}

acceleration = {
    "Terrassement": {"factor": 3.0},  # up to 5 crews
    "Fondations": {"factor": 2},    # up to 3 crews
    "Superstructure": {"factor": 1.0},    # allow at most 2
    "default": {"factor": 1.0},
}

cross_floor_links = {
    "2.1": ["1.2"],
    "4.1": ["4.7"],
    "4.2": ["4.7"],
    "4.3": ["4.7"],  # Columns(F+1) depend on Slab(F)
    "4.8": ["4.7"],
    "5.1": ["4.7"],  # Masonry(F) depends on Slab(F) (cross-floor carryover)
    # Waterproofing(F) depends on Masonry(F-1) if needed
    # Add more as project requires
}
SHIFT_CONFIG = {
    "default": 1.0,       # fallback if discipline not specified
    "Terrassement": 2.0,      # concrete works use two shifts
    "GrosOeuvres": 1.5,    # e.g., extended hours, not full 2 shifts
    "SecondeOeuvres": 1.0,     # normal single shift
   }   # maybe structure works are also two shifts


zone_floors = {"A": 4,"B": 13,}

def run_schedule(zone_floors, quantity_matrix, start_date, holidays=None):
    """
    Wraps your scheduling logic. Returns dict of DataFrames.
    """
    from reporting import BasicReporter
    tasks = generate_tasks(BASE_TASKS, zone_floors,cross_floor_links=cross_floor_links)
    # 🔍 Call validate with all arguments
    validate_tasks(tasks, workers, equipment, quantity_matrix)
    # 🔍 Apply validation & patching
    tasks, workers, equipment, quantity_matrix = validate_tasks(tasks, workers, equipment, quantity_matrix)
    workweek=[0,1,2,3,4,5]
    cal = AdvancedCalendar(start_date=start_date,holidays=holidays,workweek=workweek)  # customize holidays & workweek if needed
    dur_calc = DurationCalculator(workers, equipment, quantity_matrix)
    sched = AdvancedScheduler(tasks, workers, equipment, cal, dur_calc)
    schedule = sched.generate()
    
    reporter = BasicReporter(tasks, schedule, sched.worker_manager, sched.equipment_manager, cal)
    output_folder = reporter.export_all()
    return schedule, output_folder

def generate_quantity_template(base_tasks, zones_floors):
    """Generates an empty Excel template for quantity input by the user."""
    records = []
    for zone, max_floor in zones_floors.items():
        for floor in range(max_floor + 1):
            for discipline, tasks in base_tasks.items():
                for task in tasks:
                    records.append({
                        "TaskID": task.id,
                        "TaskName": task.name,
                        "Zone": zone,
                        "Floor": floor,
                        "Discipline": discipline,
                        "Quantity": "",  # User fills this
                        "Unit": getattr(task, "unit", "")
                    })
    df = pd.DataFrame(records)
    temp_dir = tempfile.mkdtemp(prefix="quantity_template_")
    file_path = os.path.join(temp_dir, "quantity_template.xlsx")
    df.to_excel(file_path, index=False)
    return file_path

TASK_ID_NAME = {}
for discipline, tasks in BASE_TASKS.items():
    for task in tasks:
        TASK_ID_NAME[task.id] = task.name
def generate_worker_template(workers):
    """
    Generates an Excel template for workers with task names instead of IDs.
    """
    records = []
    for worker_name, worker in workers.items():
        for task_id, prod_rate in worker.productivity_rates.items():
            records.append({
                "TaskName": TASK_ID_NAME.get(task_id, task_id),  # lookup task name
                "WorkerType": worker.name,
                "Count": worker.count,
                "HourlyRate": worker.hourly_rate,
                "ProductivityRate": prod_rate
            })
    df = pd.DataFrame(records)

    temp_dir = tempfile.mkdtemp(prefix="worker_template_")
    file_path = os.path.join(temp_dir, "worker_template.xlsx")
    df.to_excel(file_path, index=False)
    return file_path
    
def generate_equipment_template(equipment):
    """
    Generates an Excel template for equipment with task names instead of IDs.
    """
    records = []
    for eq_name, eq in equipment.items():
        for task_id, prod_rate in eq.productivity_rates.items():
            records.append({
                "TaskName": TASK_ID_NAME.get(task_id, task_id),  # lookup task name
                "EquipmentType": eq.name,
                "Count": eq.count,
                "HourlyRate": eq.hourly_rate,
                "ProductivityRate": prod_rate
            })

    df = pd.DataFrame(records)

    temp_dir = tempfile.mkdtemp(prefix="equipment_template_")
    file_path = os.path.join(temp_dir, "equipment_template.xlsx")
    df.to_excel(file_path, index=False)
    return file_path
    
def run_schedule(zone_floors, quantity_matrix, start_date, workers_dict=None, equipment_dict=None, holidays=None):
    """
    Wraps scheduling logic.
    Uses user-provided dictionaries if uploaded; otherwise defaults.
    Returns dict of DataFrames and output folder path.
    """
    from reporting import BasicReporter

    # Use defaults if no user input
    workers_used = workers_dict if workers_dict else workers
    equipment_used = equipment_dict if equipment_dict else equipment

    # Generate tasks
    tasks = generate_tasks(BASE_TASKS, zone_floors, cross_floor_links=cross_floor_links)

    # Validate and patch tasks
    tasks, workers_used, equipment_used, quantity_matrix = validate_tasks(tasks, workers_used, equipment_used, quantity_matrix)

    # Setup calendar and duration calculator
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
    st.markdown(
        """
        Upload or define your project input files.
        - Quantities per task/zone/floor
        - Worker and equipment resources
        - Output: Schedule, Gantt, Utilizations
        """
    )

    # Define Zones & Floors
    st.subheader("🏗️ Define Project Zones & Floors")
    zones_floors = {}
    num_zones = st.number_input("Number of zones?", min_value=1, max_value=20, value=2)
    for i in range(num_zones):
        zone_name = st.text_input(f"Zone {i+1} Name", value=f"Zone_{i+1}")
        max_floor = st.number_input(f"Max floors for {zone_name}", min_value=0, max_value=100, value=5, key=f"floor_{i}")
        zones_floors[zone_name] = max_floor

    # Generate templates
    if st.button("📁 Generate Default Templates"):
        # Quantity template
        qty_file = generate_quantity_template(BASE_TASKS, zones_floors)
        with open(qty_file, "rb") as f:
            st.download_button("⬇️ Download Quantity Template", f, file_name="quantity_template.xlsx")

        # Worker template
        worker_file = generate_worker_template(workers)
        with open(worker_file, "rb") as f:
            st.download_button("⬇️ Download Worker Template", f, file_name="worker_template.xlsx")

        # Equipment template
        equip_file = generate_equipment_template(equipment)
        with open(equip_file, "rb") as f:
            st.download_button("⬇️ Download Equipment Template", f, file_name="equipment_template.xlsx")

    # Upload input files
    quantity_file = st.file_uploader("📤 Upload Quantity Matrix (Excel)", type=["xlsx"])
    worker_file = st.file_uploader("📤 Upload Worker Template (Excel)", type=["xlsx"])
    equipment_file = st.file_uploader("📤 Upload Equipment Template (Excel)", type=["xlsx"])

    start_date = st.date_input("Project Start Date", value=pd.Timestamp.today())

    if st.button("🚀 Generate Schedule"):
        if not quantity_file:
            st.warning("Please upload the Quantity Matrix Excel.")
            return

        # Read quantity matrix
        quantity_matrix = pd.read_excel(quantity_file)

        # Override defaults if uploaded
        workers_used = None
        if worker_file:
            df_workers = pd.read_excel(worker_file)
            workers_used = parse_worker_excel(df_workers)  # function to convert uploaded file to workers dict

        equipment_used = None
        if equipment_file:
            df_equip = pd.read_excel(equipment_file)
            equipment_used = parse_equipment_excel(df_equip)  # function to convert uploaded file to equipment dict

        with st.spinner("Generating schedule..."):
            schedule, output_folder = run_schedule(
                zones_floors, quantity_matrix, start_date, workers_used, equipment_used
            )

        st.success("✅ Schedule generated successfully!")
        st.info("All outputs are in the temporary folder. Download immediately.")

        # Provide download links for outputs
        for file_name in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file_name)
            with open(file_path, "rb") as f:
                st.download_button(f"⬇️ {file_name}", f, file_name=file_name)

def analyze_project_progress(reference_path, actual_path):
    ref_df = pd.read_excel(reference_path, sheet_name="Schedule")
    act_df = pd.read_excel(actual_path)

    ref_df["Start"] = pd.to_datetime(ref_df["Start"])
    ref_df["End"] = pd.to_datetime(ref_df["End"])
    act_df["Date"] = pd.to_datetime(act_df["Date"])

    ref_df["PlannedDuration"] = (ref_df["End"] - ref_df["Start"]).dt.days + 1
    timeline = pd.date_range(ref_df["Start"].min(), ref_df["End"].max(), freq="D")

    planned_curve = []
    for day in timeline:
        ongoing = ref_df[(ref_df["Start"] <= day) & (ref_df["End"] >= day)]
        progress = len(ongoing) / len(ref_df)
        planned_curve.append({"Date": day, "PlannedProgress": progress})

    planned_df = pd.DataFrame(planned_curve)

    actual_df = act_df.groupby("Date", as_index=False)["Progress"].mean()
    actual_df["CumulativeActual"] = actual_df["Progress"].cumsum()
    actual_df["CumulativeActual"] = actual_df["CumulativeActual"].clip(upper=1.0)

    analysis_df = pd.merge(planned_df, actual_df, on="Date", how="outer").fillna(method="ffill")
    analysis_df["ProgressDeviation"] = analysis_df["CumulativeActual"] - analysis_df["PlannedProgress"]
    return analysis_df
def monitor_project_ui():
    st.header("📊 Project Monitoring")

    reference_file = st.file_uploader("Upload Reference Schedule (Excel)", type=["xlsx"])
    actual_file = st.file_uploader("Upload Actual Progress (Excel)", type=["xlsx"])
    from reporting import MonitoringReporter
    if reference_file and actual_file:
        ref_df = pd.read_excel(reference_file)
        act_df = pd.read_excel(actual_file)

        reporter = MonitoringReporter(ref_df, act_df)
        reporter.compute_analysis()

        st.subheader("📈 S-Curve")
        st.plotly_chart(reporter.generate_scurve(), use_container_width=True)

        st.subheader("📊 Deviation Chart")
        st.plotly_chart(reporter.generate_deviation_chart(), use_container_width=True)

        st.download_button(
            label="📥 Download Analysis CSV",
            data=reporter.analysis_df.to_csv(index=False).encode('utf-8'),
            file_name="progress_analysis.csv",
            mime="text/csv"
        )
