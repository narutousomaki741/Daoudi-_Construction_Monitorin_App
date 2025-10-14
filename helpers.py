import os
import tempfile
import pandas as pd
from typing import List,Dict, Optional
from collections import defaultdict, deque
import tempfile
from datetime import timedelta,datetime
from dataclasses import dataclass, field
from collections import defaultdict, deque
import bisect
import math
import warnings
import logging
import loguru
from models import WorkerResource, EquipmentResource, BaseTask,Task
from defaults import BASE_TASKS,acceleration, workers as default_workers, equipment as default_equipment

# Build TASK_ID_NAME from BASE_TASKS
TASK_ID_NAME = {task.id: task.name for tasks in BASE_TASKS.values() for task in tasks}
ground_disciplines=["Préliminaire","Terrassement","Fondations"]

# ------------------------- Parse Functions -------------------------


def parse_worker_excel(df: pd.DataFrame) -> Dict[str, WorkerResource]:
    """
    Parse an uploaded worker Excel file into WorkerResource objects.
    Expected columns: WorkerType, Count, HourlyRate, ProductivityRate, TaskName,TaskID, MaxCrews
    """
    workers_dict = {}

    # First pass: collect all data for each worker type
    worker_data = {}
    
    for _, row in df.iterrows():
        worker_type = str(row.get("WorkerType", "")).strip()
        if not worker_type:
            continue

        count = int(row.get("Count", 0))
        hourly_rate = float(row.get("HourlyRate", 0))

        # Parse productivity: map TaskName back to TaskID
        task_name = str(row.get("TaskName", "")).strip()
        prod_rate = float(row.get("ProductivityRate", 0))
        max_crews = int(row.get("MaxCrews", 1))
        task_id = str(row.get("TaskID", "")).strip()
        
        # Store data for this worker type
        if worker_type not in worker_data:
            worker_data[worker_type] = {
                'count': count,
                'hourly_rate': hourly_rate,
                'productivity_rates': {},
                'max_crews': {}
            }
        
        # Add task-specific data for BOTH productivity_rates and max_crews
        worker_data[worker_type]['productivity_rates'][task_id] = prod_rate
        worker_data[worker_type]['max_crews'][task_id] = max_crews

    # Second pass: create WorkerResource objects
    for worker_type, data in worker_data.items():
        workers_dict[worker_type] = WorkerResource(
            name=worker_type,
            count=data['count'],
            hourly_rate=data['hourly_rate'],
            productivity_rates=data['productivity_rates'],  # Complete dictionary
            skills=[worker_type],
            max_crews=data['max_crews']  # Complete dictionary
        )

    return workers_dict if workers_dict else default_workers

def parse_equipment_excel(df: pd.DataFrame) -> Dict[str, EquipmentResource]:
    """
    Parse an uploaded equipment Excel file into EquipmentResource objects.
    Expected columns: EquipmentType, Count, HourlyRate, ProductivityRate, TaskName,TaskID, MaxEquipment
    """
    equipment_dict = {}
    
    # First pass: collect all data for each equipment type
    equipment_data = {}

    for _, row in df.iterrows():
        eq_type = str(row.get("EquipmentType", "")).strip()
        if not eq_type:
            continue

        count = int(row.get("Count", 0))
        hourly_rate = float(row.get("HourlyRate", 0))

        # Parse productivity: map TaskName back to TaskID
        task_name = str(row.get("TaskName", "")).strip()
        prod_rate = float(row.get("ProductivityRate", 0))
        max_equipment = int(row.get("MaxEquipment", 1))  # NEW: Parse MaxEquipment
        task_id = str(row.get("TaskID", "")).strip()
        
        # Store data for this equipment type
        if eq_type not in equipment_data:
            equipment_data[eq_type] = {
                'count': count,
                'hourly_rate': hourly_rate,
                'productivity_rates': {},
                'max_equipment': {}
            }
        
        # Add task-specific data
        equipment_data[eq_type]['productivity_rates'][task_id] = prod_rate
        equipment_data[eq_type]['max_equipment'][task_id] = max_equipment

    # Second pass: create EquipmentResource objects
    for eq_type, data in equipment_data.items():
        equipment_dict[eq_type] = EquipmentResource(
            name=eq_type,
            count=data['count'],
            hourly_rate=data['hourly_rate'],
            productivity_rates=data['productivity_rates'],
            max_equipment=data['max_equipment'],  # Now a dictionary
            type="general"
        )

    return equipment_dict if equipment_dict else default_equipment

def parse_quantity_excel(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Parse a quantity Excel uploaded by the user.
    Expected columns: TaskID, Zone, Floor, Quantity
    Returns a nested dictionary: {task_id: {floor: {zone: quantity}}}
    """
    quantity_matrix = {}
    for _, row in df.iterrows():
        try:
            task_id = str(row.get("TaskID", "")).strip()
            if not task_id:
                continue
            zone = str(row.get("Zone", "")).strip()
            if not zone:
                continue
            # Safely convert floor to integer
            try:
                floor = int(float(row.get("Floor", 0)))
            except (ValueError, TypeError):
                continue     
            # Safely convert quantity to float
            try:
                quantity = float(row.get("Quantity", 0) or 0)
            except (ValueError, TypeError):
                quantity = 0.0 
            # Build the nested dictionary structure
            if task_id not in quantity_matrix:
                quantity_matrix[task_id] = {}
            if floor not in quantity_matrix[task_id]:
                quantity_matrix[task_id][floor] = {}
            quantity_matrix[task_id][floor][zone] = quantity
        except Exception as e:
            print(f"Warning: Skipping row due to error: {e}")
            continue
    return quantity_matrix

# ------------------------- Template Generation -------------------------

def generate_worker_template(workers_dict=default_workers):
    """
    Generates an Excel template for workers with task names instead of IDs.
    """
    records = []
    for worker_name, worker in workers_dict.items():
        for task_id, prod_rate in worker.productivity_rates.items():
            # Get max_crews for this specific task
            max_crews = 1  # default
            if hasattr(worker, 'max_crews'):
                if isinstance(worker.max_crews, dict):
                    max_crews = worker.max_crews.get(task_id, 1)
                else:
                    max_crews = worker.max_crews if worker.max_crews else 20
            
            records.append({
                 "TaskID":task_id,
                "TaskName": TASK_ID_NAME.get(task_id, task_id),  # lookup task name
                "WorkerType": worker.name,
                "Count": worker.count,
                "HourlyRate": worker.hourly_rate,
                "ProductivityRate": prod_rate,
                "MaxCrews": max_crews  # NEW: Add max_crews column
            })
    df = pd.DataFrame(records)
    temp_dir = tempfile.mkdtemp(prefix="worker_template_")
    file_path = os.path.join(temp_dir, "worker_template.xlsx")
    df.to_excel(file_path, index=False)
    return file_path
    
def generate_equipment_template(equipment_dict=default_equipment):
    """
    Generates an Excel template for equipment with task names instead of IDs.
    """
    records = []
    for eq_name, eq in equipment_dict.items():
        for task_id, prod_rate in eq.productivity_rates.items():
            # Get max_equipment for this specific task
            max_equipment = 1  # default
            if hasattr(eq, 'max_equipment'):
                if isinstance(eq.max_equipment, dict):
                    max_equipment = eq.max_equipment.get(task_id, 1)
                else:
                    max_equipment = eq.max_equipment if eq.max_equipment else 1
            
            records.append({
                "TaskID": task_id,
                "TaskName": TASK_ID_NAME.get(task_id, task_id),  # lookup task name
                "EquipmentType": eq.name,
                "Count": eq.count,
                "HourlyRate": eq.hourly_rate,
                "ProductivityRate": prod_rate,
                "MaxEquipment": max_equipment  # NEW: Add MaxEquipment column
            })
    df = pd.DataFrame(records)
    temp_dir = tempfile.mkdtemp(prefix="equipment_template_")
    file_path = os.path.join(temp_dir, "equipment_template.xlsx")
    df.to_excel(file_path, index=False)
    return file_path

def generate_quantity_template(base_tasks=BASE_TASKS, zones_floors=None):
    """Generates an empty Excel template for quantity input by the user."""
    if zones_floors is None:
        zones_floors = {"Zone1": 0}  # default fallback
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
        
        if isinstance(res_max_value, Dict):
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
