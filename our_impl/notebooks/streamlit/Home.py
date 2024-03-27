import streamlit as st
import pandas as pd
import numpy as np

from utils import utils


def local_css(file_name):
    with open(file_name) as f:
        css = f.read()

    return css


st.set_page_config(
    page_title="Promotion optimizer", layout="wide", initial_sidebar_state="expanded"
)


st.title("Promotion optimization for retail stores")


st.subheader("How to use this application")
st.markdown(
    """
    1. Upload your store datasets
    2. Optimize 
    """
)

utils.add_logo()

st.subheader("Sample dataset format for store")
sample_data = pd.read_csv("../assets/combined_milk_final.csv")
st.write(sample_data)
csv = sample_data.to_csv(index=False)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="sample_data.csv",
    mime="text/csv",
)

css = local_css("./style.css")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
