# utils.py
import pandas as pd
import streamlit as st
import plotly.express as px

def load_excel(file):
    """Reads an Excel file into a DataFrame safely."""
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {e}")
        return None

def display_gantt(df, title="Project Gantt Chart"):
    """Displays Gantt chart using Plotly."""
    try:
        fig = px.timeline(
            df,
            x_start="Start",
            x_end="End",
            y="TaskName",
            color="Discipline",
            hover_name="TaskName",
            title=title
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Unable to display Gantt chart: {e}")

def display_s_curve(df, title="S-Curve Progress"):
    """Displays an S-curve of cumulative progress."""
    try:
        fig = px.line(
            df,
            x="WeekStart",
            y="CumulativeProgress",
            color="Discipline",
            title=title,
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not plot S-curve: {e}")

def download_file(file_path, label):
    """Allow user to download a file from a path."""
    with open(file_path, "rb") as f:
        st.download_button(label, f, file_name=file_path.split("/")[-1])