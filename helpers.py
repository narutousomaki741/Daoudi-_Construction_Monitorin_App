import os
import tempfile
import pandas as pd
from typing import Dict
from models import WorkerResource, EquipmentResource, BaseTask
from defaults import BASE_TASKS, workers as default_workers, equipment as default_equipment

# Build TASK_ID_NAME from BASE_TASKS
TASK_ID_NAME = {task.id: task.name for tasks in BASE_TASKS.values() for task in tasks}


# ------------------------- Parse Functions -------------------------


def parse_worker_excel(df: pd.DataFrame) -> Dict[str, WorkerResource]:
    """
    Parse an uploaded worker Excel file into WorkerResource objects.
    Expected columns: WorkerType, Count, HourlyRate, ProductivityRate, TaskName, MaxCrews
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
        task_id = next((tid for tid, tname in TASK_ID_NAME.items() if tname == task_name), task_name)
        
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
    Expected columns: EquipmentType, Count, HourlyRate, ProductivityRate, TaskName
    """
    equipment_dict = {}

    for _, row in df.iterrows():
        eq_type = str(row.get("EquipmentType", "")).strip()
        if not eq_type:
            continue

        count = int(row.get("Count", 0))
        hourly_rate = float(row.get("HourlyRate", 0))

        # Parse productivity: map TaskName back to TaskID
        task_name = str(row.get("TaskName", "")).strip()
        prod_rate = float(row.get("ProductivityRate", 0))
        task_id = next((tid for tid, tname in TASK_ID_NAME.items() if tname == task_name), task_name)
        
        # Initialize or update EquipmentResource
        if eq_type not in equipment_dict:
            equipment_dict[eq_type] = EquipmentResource(
                name=eq_type,
                count=count,
                hourly_rate=hourly_rate,
                productivity_rates={task_id: prod_rate},
                max_equipment=1,
                type="general"
            )
        else:
            equipment_dict[eq_type].productivity_rates[task_id] = prod_rate

    return equipment_dict if equipment_dict else default_equipment


def parse_quantity_excel(df: pd.DataFrame) -> Dict[str, float]:
    """
    Parse a quantity Excel uploaded by the user.
    Expected columns: TaskID, Zone, Floor, Quantity
    Returns a dictionary keyed by TaskID+Zone+Floor
    """
    quantity_dict = {}
    for _, row in df.iterrows():
        task_id = str(row.get("TaskID", "")).strip()
        zone = str(row.get("Zone", "")).strip()
        floor = int(row.get("Floor", 0))
        quantity = float(row.get("Quantity", 0) or 0)
        key = f"{task_id}_{zone}_{floor}"
        quantity_dict[key] = quantity
    return quantity_dict


# ------------------------- Template Generation -------------------------

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
