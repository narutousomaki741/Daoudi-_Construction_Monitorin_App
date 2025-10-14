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
from models import Task,BaseTask, WorkerResource, EquipmentResource
from defaults import workers, equipment, BASE_TASKS, cross_floor_links, acceleration, SHIFT_CONFIG
from logic import (
    AdvancedCalender,
    DurationCalculator,
    CPMAnalyzer,
    AdvancedScheduler
          )
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


# -----------------------------
# Calendar (workdays) - half-open end
# -----------------------------


def generate_schedule_ui():
    from reporting import generate_interactive_gantt
    """Main Streamlit interface for Construction Scheduling Web App"""
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
                help="A zone is a distinct section of your building",
            )

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
                    max_floor = st.number_input(
                        "Floors", min_value=0, max_value=60, value=5, key=f"floor_{i}"
                    )
                    zones_floors[zone_name] = max_floor

        start_date = st.date_input("Project Start Date", value=pd.Timestamp.today())

        with st.expander("ℹ️ Project Information", expanded=False):
            project_name = st.text_input("Project Name", value="My Construction Project")
            project_manager = st.text_input(
                "Project Manager", placeholder="Enter project manager name"
            )

    # ---------------- TAB 2: TEMPLATE DOWNLOADS ----------------
    with tab2:
        st.subheader("📊 Generate Default Data Templates")
        st.markdown("""
        **Step 1:** Download and fill templates with your project data:
        - Quantity Template → task quantities per zone/floor  
        - Worker Template → crew sizes and productivity rates  
        - Equipment Template → machine counts and rates
        """)

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

        # 🔁 Always show download buttons once generated
        if st.session_state.get("templates_ready", False):
            st.subheader("⬇️ Download Templates")
            col1, col2, col3 = st.columns(3)
            for label, key, col in [
                ("⬇️ Quantity Template", "qty_file", col1),
                ("⬇️ Worker Template", "worker_file", col2),
                ("⬇️ Equipment Template", "equip_file", col3),
            ]:
                path = st.session_state.get(key)
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        col.download_button(
                            label,
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

        # Run scheduling engine only once
        if st.button("🚀 Generate Project Schedule", type="primary", use_container_width=True, disabled=not all_ready):
            try:
                progress = st.progress(0)
                status = st.empty()

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

                # Store generated files in session
                st.session_state["schedule_generated"] = True
                st.session_state["output_folder"] = output_folder
                st.session_state["generated_files"] = [
                    os.path.join(output_folder, f) for f in os.listdir(output_folder)
                ]

                # Generate Gantt chart HTML
                schedule_excel_path = os.path.join(output_folder, "construction_schedule_optimized.xlsx")
                if os.path.exists(schedule_excel_path):
                    gantt_html = os.path.join(output_folder, "interactive_gantt.html")
                    generate_interactive_gantt(pd.read_excel(schedule_excel_path), gantt_html)
                    st.session_state["generated_files"].append(gantt_html)

                st.success("🎉 Schedule generated successfully!")
                progress.progress(100)

            except Exception as e:
                st.error(f"❌ Failed to generate schedule: {e}")
                if st.checkbox("🔍 Show error details"):
                    st.exception(e)

        # ---------------- DOWNLOAD SECTION (always visible) ----------------
        if st.session_state.get("schedule_generated", False):
            st.subheader("📂 Download Generated Excel Files")

            excel_files = [f for f in st.session_state["generated_files"] if f.endswith(".xlsx")]
            cols = st.columns(3)
            for i, file_path in enumerate(excel_files):
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        cols[i % 3].download_button(
                            os.path.basename(file_path),
                            f,
                            file_name=os.path.basename(file_path),
                            use_container_width=True,
                        )

            # ---- GANTT CHART (separate orange button) ----
            gantt_files = [f for f in st.session_state["generated_files"] if f.endswith(".html")]
            if gantt_files:
                st.subheader("📊 Interactive Gantt Chart")

                st.markdown("""
                    <style>
                    div.stButton > button.custom-gantt {
                        background-color: #ff7f0e;
                        color: white;
                        border-radius: 12px;
                        padding: 0.6em 1.2em;
                        border: none;
                        font-weight: bold;
                        font-size: 16px;
                        width: 40%;
                        display: block;
                        margin: 0 auto;
                        transition: 0.3s;
                    }
                    div.stButton > button.custom-gantt:hover {
                        background-color: #e06a00;
                        transform: scale(1.03);
                    }
                    </style>
                """, unsafe_allow_html=True)

                gantt_file = gantt_files[0]
                if os.path.exists(gantt_file):
                    with open(gantt_file, "rb") as f:
                        st.download_button(
                            "⬇️ Download Interactive Gantt (HTML)",
                            f,
                            file_name=os.path.basename(gantt_file),
                            key="gantt_download",
                            use_container_width=True,
                        )

                # Apply CSS class via JS (to style button)
                st.markdown("""
                    <script>
                    const btns = window.parent.document.querySelectorAll('button[kind="secondary"]');
                    if (btns.length) {
                        btns[btns.length - 1].classList.add('custom-gantt');
                    }
                    </script>
                """, unsafe_allow_html=True)

    # ---------------- SIDEBAR HELP ----------------
    with st.sidebar:
        st.header("💡 Help & Guidance")
        st.markdown("""
        **Steps:**
        1️⃣ Configure project zones & floors  
        2️⃣ Download Excel templates  
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
        
