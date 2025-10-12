import pandas as pd
from collections import defaultdict

# ---------------- Quantity Matrix Parser ----------------
def parse_quantity_excel(df):
    """
    Converts uploaded quantity Excel into a dictionary:
    quantity_matrix[zone][floor][task_id] = quantity
    """
    quantity_matrix = defaultdict(lambda: defaultdict(dict))
    
    for _, row in df.iterrows():
        zone = row.get("Zone")
        floor = row.get("Floor")
        task_id = row.get("TaskID")
        qty = row.get("Quantity", 1)  # default to 1 if empty

        # Ensure numeric
        try:
            qty = float(qty)
        except (ValueError, TypeError):
            qty = 1

        quantity_matrix[zone][floor][task_id] = qty

    return quantity_matrix

# ---------------- Worker Excel Parser ----------------
def parse_worker_excel(df):
    """
    Converts uploaded worker Excel into dictionary:
    workers[worker_name] = WorkerObject
    Expects columns: TaskName, WorkerType, Count, HourlyRate, ProductivityRate
    """
    from defaults import workers as Worker  # Make sure your Worker class is imported

    workers = {}

    for _, row in df.iterrows():
        worker_type = row["WorkerType"]
        task_name = row["TaskName"]
        count = row.get("Count", 1)
        rate = row.get("HourlyRate", 0)
        prod = row.get("ProductivityRate", 1)

        if worker_type not in workers:
            workers[worker_type] = Worker(name=worker_type, count=count, hourly_rate=rate, productivity_rates={})

        workers[worker_type].productivity_rates[task_name] = prod

    return workers

# ---------------- Equipment Excel Parser ----------------
def parse_equipment_excel(df):
    """
    Converts uploaded equipment Excel into dictionary:
    equipment[equipment_name] = EquipmentObject
    Expects columns: TaskName, EquipmentType, Count, HourlyRate, ProductivityRate
    """
    from defaults import equipment as Equipment  # Make sure your Equipment class is imported

    equipment = {}

    for _, row in df.iterrows():
        eq_type = row["EquipmentType"]
        task_name = row["TaskName"]
        count = row.get("Count", 1)
        rate = row.get("HourlyRate", 0)
        prod = row.get("ProductivityRate", 1)

        if eq_type not in equipment:
            equipment[eq_type] = Equipment(name=eq_type, count=count, hourly_rate=rate, productivity_rates={})

        equipment[eq_type].productivity_rates[task_name] = prod

    return equipment
