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

# Import your existing modules (UNCHANGED)
from models import Task, BaseTask, WorkerResource, EquipmentResource
from defaults import workers, equipment, BASE_TASKS, cross_floor_links, acceleration, SHIFT_CONFIG
from logic import run_schedule
from helpers import (
    parse_quantity_excel, parse_worker_excel, parse_equipment_excel,
    generate_quantity_template, generate_worker_template, generate_equipment_template
)
from reporting import generate_interactive_gantt
from logic imprt Analyse_project_progress
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ground_disciplines = ["Préliminaire", "Terrassement", "Fondations"]

def inject_ui_styles():
    """Inject professional UI styles"""
    st.markdown("""
    <style>
    /* Enhanced tab styling */
    .enhanced-tab {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    
    /* File upload styling */
    .uploaded-file {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    
    /* Status indicators */
    .status-ready { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button group styling */
    .button-group {
        display: flex;
        gap: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_metric_row(metrics_dict):
    """Create a row of professional metric cards"""
    cols = st.columns(len(metrics_dict))
    for idx, (title, value) in enumerate(metrics_dict.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 0.9rem; opacity: 0.9;">{title}</div>
                <div style="font-size: 1.8rem; font-weight: bold;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

def create_info_card(title, content, icon="ℹ️", card_type="info"):
    """Create professional information cards"""
    colors = {
        "info": "#1f77b4",
        "success": "#28a745", 
        "warning": "#ffc107",
        "error": "#dc3545"
    }
    
    st.markdown(f"""
    <div class="professional-card" style="border-left-color: {colors[card_type]};">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <h4 style="margin: 0; color: {colors[card_type]};">{title}</h4>
        </div>
        <div style="color: #555;">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def render_upload_section(title, key_suffix, accepted_types=["xlsx"]):
    """Enhanced file upload section"""
    with st.container():
        st.markdown(f"**{title}**")
        
        uploaded_file = st.file_uploader(
            f"Upload {title}",
            type=accepted_types,
            key=f"upload_{key_suffix}",
            help=f"Upload your {title} file in Excel format"
        )
        
        if uploaded_file:
            file_size = uploaded_file.size / 1024 / 1024  # Convert to MB
            file_details = {
                "name": uploaded_file.name,
                "size": f"{file_size:.2f} MB",
                "type": uploaded_file.type
            }
            
            st.markdown(f"""
            <div class="uploaded-file">
                <strong>✅ {uploaded_file.name}</strong><br>
                <small>Size: {file_size:.2f} MB | Type: {uploaded_file.type}</small>
            </div>
            """, unsafe_allow_html=True)
            
            if file_size < 0.1:  # 100KB
                st.warning("⚠️ File size is very small. Please verify the content.")
            elif file_size > 50:
                st.error("❌ File size exceeds 50MB limit.")
                
        return uploaded_file

def generate_schedule_ui():
    """Enhanced main Streamlit interface for Construction Scheduling Web App"""
    inject_ui_styles()
    
    # Header with metrics
    st.markdown('<div class="main-header">🏗️ Construction Project Scheduler Pro</div>', unsafe_allow_html=True)
    
    # Quick metrics
    if st.session_state.get("schedule_generated"):
        create_metric_row({
            "Zones Configured": f"{len(st.session_state.get('zones_floors', {}))}",
            "Tasks Processed": "Calculating...",
            "Schedule Duration": "Calculating...",
            "Files Generated": f"{len(st.session_state.get('generated_files', []))}"
        })
    
    # Enhanced tabs with better icons and spacing
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Project Setup", 
        "📁 Templates", 
        "📤 Upload Data", 
        "🚀 Generate & Results"
    ])

    # ---------------- TAB 1: ENHANCED PROJECT SETUP ----------------
    with tab1:
        st.subheader("🏗️ Project Configuration")
        
        create_info_card(
            "Project Setup Guide",
            "Configure your building zones, floors, and project timeline. Each zone represents a distinct section of your building.",
            "🏗️",
            "info"
        )
        
        with st.expander("🏢 Building Configuration", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                num_zones = st.number_input(
                    "How many zones does your building have?",
                    min_value=1,
                    max_value=30,
                    value=2,
                    help="A zone is a distinct section of your building (e.g., North Wing, Tower A)",
                    key="num_zones_input"
                )
            
            with col2:
                st.metric("Zones Configured", num_zones)
            
            zones_floors = {}
            st.markdown("### Zone Details")
            
            for i in range(num_zones):
                with st.container():
                    st.markdown(f"**Zone {i + 1}**")
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        zone_name = st.text_input(
                            "Zone name",
                            value=f"Zone_{i + 1}",
                            key=f"zone_name_{i}",
                            placeholder="e.g., North_Wing, Tower_A"
                        )
                    
                    with col2:
                        max_floor = st.number_input(
                            "Floors",
                            min_value=0,
                            max_value=60,
                            value=5,
                            key=f"floor_{i}"
                        )
                    
                    with col3:
                        st.metric("Floors", max_floor)
                    
                    zones_floors[zone_name] = max_floor
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Project Start Date", 
                    value=pd.Timestamp.today(),
                    help="Select the planned start date for your project"
                )
            
            with col2:
                st.metric("Start Date", start_date.strftime("%Y-%m-%d"))
        
        with st.expander("📝 Project Information", expanded=False):
            project_name = st.text_input(
                "Project Name", 
                value="My Construction Project",
                placeholder="Enter a descriptive project name"
            )
            project_manager = st.text_input(
                "Project Manager",
                placeholder="Enter project manager name"
            )
            
            if project_name and project_manager:
                st.success(f"✅ Project '{project_name}' configured with manager {project_manager}")

        # Store in session state
        st.session_state["zones_floors"] = zones_floors
        st.session_state["start_date"] = start_date

    # ---------------- TAB 2: ENHANCED TEMPLATE DOWNLOADS ----------------
    with tab2:
        st.subheader("📊 Generate Data Templates")
        
        create_info_card(
            "Template Instructions",
            "Download these templates, fill them with your project data, then upload in the next tab. Each template serves a specific purpose in the scheduling process.",
            "📋",
            "info"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_info_card(
                "Quantity Template",
                "Task quantities per zone/floor - defines the scope of work",
                "📏",
                "info"
            )
        
        with col2:
            create_info_card(
                "Worker Template", 
                "Crew sizes and productivity rates - defines labor resources",
                "👷",
                "info"
            )
        
        with col3:
            create_info_card(
                "Equipment Template",
                "Machine counts and rates - defines equipment resources",
                "🚜",
                "info"
            )
        
        if st.button("🎯 Generate All Templates", type="primary", use_container_width=True):
            try:
                with st.spinner("🔄 Preparing professional templates..."):
                    qty_file = generate_quantity_template(BASE_TASKS, st.session_state.get("zones_floors", {}))
                    worker_file = generate_worker_template(workers)
                    equip_file = generate_equipment_template(equipment)
                    
                    st.session_state["templates_ready"] = True
                    st.session_state["qty_file"] = qty_file
                    st.session_state["worker_file"] = worker_file
                    st.session_state["equip_file"] = equip_file
                    
                    st.success("✅ All templates generated successfully!")
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Failed to generate templates: {e}")
                logger.error(f"Template generation error: {e}")

        # Enhanced download section
        if st.session_state.get("templates_ready", False):
            st.markdown("---")
            st.subheader("⬇️ Download Templates")
            
            templates_info = [
                ("📏 Quantity Template", "qty_file", "Defines task quantities across zones/floors"),
                ("👷 Worker Template", "worker_file", "Specifies crew sizes and productivity rates"),
                ("🚜 Equipment Template", "equip_file", "Details machine counts and operational rates")
            ]
            
            for icon, key, description in templates_info:
                path = st.session_state.get(key)
                if path and os.path.exists(path):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{icon} {description}**")
                        with col2:
                            with open(path, "rb") as f:
                                st.download_button(
                                    "Download",
                                    f,
                                    file_name=os.path.basename(path),
                                    use_container_width=True,
                                    key=f"download_{key}"
                                )

    # ---------------- TAB 3: ENHANCED UPLOAD ----------------
    with tab3:
        st.subheader("📤 Upload Your Project Data")
        
        create_info_card(
            "Upload Requirements",
            "Upload all three filled templates to proceed with schedule generation. Ensure data is complete and formatted correctly.",
            "📤",
            "info"
        )
        
        # Enhanced upload sections
        quantity_file = render_upload_section("Quantity Matrix", "quantity")
        worker_file = render_upload_section("Worker Template", "worker") 
        equipment_file = render_upload_section("Equipment Template", "equipment")
        
        # Upload status
        upload_status = {
            "Quantity Matrix": bool(quantity_file),
            "Worker Template": bool(worker_file),
            "Equipment Template": bool(equipment_file)
        }
        
        st.markdown("### 📊 Upload Status")
        status_cols = st.columns(3)
        for idx, (name, status) in enumerate(upload_status.items()):
            with status_cols[idx]:
                if status:
                    st.success(f"✅ {name}")
                else:
                    st.warning(f"⏳ {name}")

    # ---------------- TAB 4: ENHANCED SCHEDULE GENERATION ----------------
    with tab4:
        st.subheader("🚀 Generate Project Schedule")
        
        # Upload status check
        all_ready = all([quantity_file, worker_file, equipment_file])
        
        if all_ready:
            create_info_card(
                "Ready to Generate",
                "All required files have been uploaded. Click the button below to generate your optimized construction schedule.",
                "✅",
                "success"
            )
            
            # Enhanced generate button
            if st.button("🚀 Generate Optimized Schedule", type="primary", use_container_width=True):
                try:
                    # Enhanced progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    steps = [
                        "📊 Parsing Excel files...",
                        "🔍 Validating data...", 
                        "🏗️ Running schedule engine...",
                        "📈 Generating reports...",
                        "🎉 Finalizing output..."
                    ]
                    
                    for i, step in enumerate(steps):
                        status_text.subheader(step)
                        progress_bar.progress((i + 1) * 20)
                        time.sleep(0.5)  # Simulate processing
                        
                        if i == 0:  # Parsing
                            df_quantity = pd.read_excel(quantity_file)
                            quantity_used = parse_quantity_excel(df_quantity)
                            
                        elif i == 1:  # Validation
                            df_worker = pd.read_excel(worker_file)
                            workers_used = parse_worker_excel(df_worker)
                            
                        elif i == 2:  # Scheduling
                            df_equip = pd.read_excel(equipment_file)
                            equipment_used = parse_equipment_excel(df_equip)
                            
                        elif i == 3:  # Engine
                            schedule, output_folder = run_schedule(
                                zone_floors=st.session_state.get("zones_floors", {}),
                                quantity_matrix=quantity_used,
                                start_date=st.session_state.get("start_date"),
                                workers_dict=workers_used or workers,
                                equipment_dict=equipment_used or equipment,
                                holidays=None,
                            )
                            
                        elif i == 4:  # Reporting
                            st.session_state["schedule_generated"] = True
                            st.session_state["output_folder"] = output_folder
                            st.session_state["generated_files"] = [
                                os.path.join(output_folder, f) for f in os.listdir(output_folder)
                            ]
                            
                            # Generate Gantt chart
                            schedule_excel_path = os.path.join(output_folder, "construction_schedule_optimized.xlsx")
                            if os.path.exists(schedule_excel_path):
                                gantt_html = os.path.join(output_folder, "interactive_gantt.html")
                                generate_interactive_gantt(pd.read_excel(schedule_excel_path), gantt_html)
                                st.session_state["generated_files"].append(gantt_html)
                    
                    progress_bar.progress(100)
                    status_text.subheader("✅ Schedule Generated Successfully!")
                    st.balloons()
                    
                    # Show success metrics
                    create_metric_row({
                        "Total Tasks": f"{len(schedule) if 'schedule' in locals() else 'N/A'}",
                        "Project Duration": "Calculated",
                        "Output Files": f"{len(st.session_state.get('generated_files', []))}",
                        "Status": "✅ Complete"
                    })
                    
                except Exception as e:
                    st.error(f"❌ Schedule generation failed: {e}")
                    logger.error(f"Schedule generation error: {e}")
                    
                    with st.expander("🔧 Technical Details"):
                        st.exception(e)
        else:
            missing = [name for name, status in upload_status.items() if not status]
            create_info_card(
                "Action Required",
                f"Please upload the following files: {', '.join(missing)}",
                "⚠️",
                "warning"
            )

        # Enhanced download section
        if st.session_state.get("schedule_generated", False):
            st.markdown("---")
            st.subheader("📂 Download Results")
            
            # Excel files
            excel_files = [f for f in st.session_state["generated_files"] if f.endswith(".xlsx")]
            if excel_files:
                st.markdown("#### 📊 Excel Reports")
                cols = st.columns(3)
                for i, file_path in enumerate(excel_files):
                    if os.path.exists(file_path):
                        with cols[i % 3]:
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    f"📥 {os.path.basename(file_path)}",
                                    f,
                                    file_name=os.path.basename(file_path),
                                    use_container_width=True,
                                    key=f"excel_download_{i}"
                                )
            
            # Gantt chart
            gantt_files = [f for f in st.session_state["generated_files"] if f.endswith(".html")]
            if gantt_files:
                st.markdown("#### 📈 Interactive Gantt Chart")
                gantt_file = gantt_files[0]
                if os.path.exists(gantt_file):
                    with open(gantt_file, "rb") as f:
                        st.download_button(
                            "📊 Download Interactive Gantt Chart",
                            f,
                            file_name="project_gantt_chart.html",
                            use_container_width=True,
                            type="secondary",
                            key="gantt_download"
                        )

    # ---------------- ENHANCED SIDEBAR HELP ----------------
    with st.sidebar:
        st.markdown("---")
        st.subheader("💡 Professional Guidance")
        
        with st.expander("🚀 Quick Start Guide"):
            st.markdown("""
            1. **Configure** your building zones and floors
            2. **Download** and fill the Excel templates  
            3. **Upload** your completed data files
            4. **Generate** the optimized schedule
            5. **Download** and analyze the results
            """)
        
        with st.expander("📋 File Requirements"):
            st.markdown("""
            - **Quantity Matrix**: Task quantities per zone/floor
            - **Worker Template**: Crew sizes and productivity rates  
            - **Equipment Template**: Machine counts and rates
            - All files must be in Excel format (.xlsx)
            """)
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            - **File upload issues**: Check file size and format
            - **Generation errors**: Verify all required columns are present
            - **Performance**: Large projects may take several minutes
            - **Support**: Contact help@constructionpro.com
            """)
  
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
        
