from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

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
