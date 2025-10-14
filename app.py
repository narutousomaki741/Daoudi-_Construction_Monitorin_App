import streamlit as st
from ui_pages import generate_schedule_ui, monitor_project_ui

st.set_page_config(page_title="Construction Project Tool", layout="wide")

st.sidebar.title("🏗 Project Management App")
mission = st.sidebar.radio(
    "Select Mission",
    ["Generate Schedule", "Monitor Project"]
)

if mission == "Generate Schedule":
    generate_schedule_ui()
elif mission == "Monitor Project":
    monitor_project_ui()
