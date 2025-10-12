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
   }
