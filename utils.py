# utils.py
import pandas as pd
import streamlit as st
import plotly.express as px

import pandas as pd
import os
import tempfile

def load_excel(file):
    """Load an uploaded Excel file into a Pandas DataFrame."""
    if file is not None:
        return pd.read_excel(file)
    return None

def save_excel_file(df, filename_prefix="output"):
    """Save DataFrame to a temporary Excel file and return the path."""
    if df is None or df.empty:
        raise ValueError("Cannot save an empty DataFrame.")
    temp_dir = tempfile.mkdtemp(prefix="construction_app_")
    path = os.path.join(temp_dir, f"{filename_prefix}.xlsx")
    df.to_excel(path, index=False)
    return path

def validate_inputs(**kwargs):
    """Check if all inputs are valid."""
    for key, value in kwargs.items():
        if value is None:
            raise ValueError(f"Missing input: {key}")

def download_file(file_path, label):
    """Allow user to download a file from a path."""
    with open(file_path, "rb") as f:
        st.download_button(label, f, file_name=file_path.split("/")[-1])