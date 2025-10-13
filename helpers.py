import pandas as pd
from typing import Dict, Optional
from defaults import WorkerResource, EquipmentResource, workers as default_workers, equipment as default_equipment, BASE_TASKS, cross_floor_links
from dataclasses import dataclass, field
from typing import List, Optional, Dict

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



# ---------------- Utility: Task ID -> Name mapping ----------------
TASK_ID_NAME = {}
for discipline, tasks in BASE_TASKS.items():
    for task in tasks:
        TASK_ID_NAME[task.id] = task.name

# ---------------- Parse Quantity Excel ----------------
def parse_quantity_excel(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Converts uploaded quantity Excel into a nested dictionary:
    quantity_matrix[task_id][zone][floor] = quantity
    """
    quantity_matrix = {}
    for _, row in df.iterrows():
        task_id = str(row.get("TaskID") or row.get("TaskName"))
        zone = row.get("Zone")
        floor = int(row.get("Floor", 0))
        qty = row.get("Quantity", 0)
        if task_id not in quantity_matrix:
            quantity_matrix[task_id] = {}
        if zone not in quantity_matrix[task_id]:
            quantity_matrix[task_id][zone] = {}
        quantity_matrix[task_id][zone][floor] = qty
    return quantity_matrix

# ---------------- Parse Worker Excel ----------------
def parse_worker_excel(df: pd.DataFrame) -> Dict[str, WorkerResource]:
    """
    Converts uploaded worker Excel into WorkerResource dictionary.
    Falls back to defaults if no file uploaded.
    """
    workers_dict = {}
    for _, row in df.iterrows():
        worker_type = row.get("WorkerType") or row.get("name")
        count = int(row.get("Count", 1))
        rate = float(row.get("HourlyRate", 0))
        productivity = {}
        task_name = row.get("TaskName")
        prod_rate = row.get("ProductivityRate", 0)
        if task_name:
            # Convert task name to task ID
            task_id = None
            for tid, tname in TASK_ID_NAME.items():
                if tname == task_name:
                    task_id = tid
                    break
            if task_id:
                productivity[task_id] = prod_rate
        workers_dict[worker_type] = WorkerResource(
            name=worker_type,
            count=count,
            hourly_rate=rate,
            productivity_rates=productivity
        )
    # Merge with defaults for missing workers
    for k, v in default_workers.items():
        if k not in workers_dict:
            workers_dict[k] = v
    return workers_dict

# ---------------- Parse Equipment Excel ----------------
def parse_equipment_excel(df: pd.DataFrame) -> Dict[str, EquipmentResource]:
    """
    Converts uploaded equipment Excel into EquipmentResource dictionary.
    Falls back to defaults if no file uploaded.
    """
    equipment_dict = {}
    for _, row in df.iterrows():
        eq_type = row.get("EquipmentType") or row.get("name")
        count = int(row.get("Count", 1))
        rate = float(row.get("HourlyRate", 0))
        productivity = {}
        task_name = row.get("TaskName")
        prod_rate = row.get("ProductivityRate", 0)
        if task_name:
            # Convert task name to task ID
            task_id = None
            for tid, tname in TASK_ID_NAME.items():
                if tname == task_name:
                    task_id = tid
                    break
            if task_id:
                productivity[task_id] = prod_rate
        equipment_dict[eq_type] = EquipmentResource(
            name=eq_type,
            count=count,
            hourly_rate=rate,
            productivity_rates=productivity
        )
    # Merge with defaults for missing equipment
    for k, v in default_equipment.items():
        if k not in equipment_dict:
            equipment_dict[k] = v
    return equipment_dict
