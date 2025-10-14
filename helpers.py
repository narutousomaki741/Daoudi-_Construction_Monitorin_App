import os
import tempfile
import pandas as pd
from typing import Dict
from collections import defaultdict, deque
from models import WorkerResource, EquipmentResource, BaseTask
from defaults import BASE_TASKS,Task, workers as default_workers, equipment as default_equipment

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
