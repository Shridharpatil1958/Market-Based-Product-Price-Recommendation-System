import streamlit as st

import pandas as pd

import sys

import os

# Add parent directory to path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.api_client import ProductPriceAPI

from utils.data_cleaner import DataCleaner

from utils.feature_engineer import FeatureEngineer

# Page config

st.set_page_config(page_title="Live Market Data", page_icon="📊", layout="wide")

# Custom CSS

st.markdown("""

<style>

.main {

background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

padding: 2rem;

}

.data-card {

background: white;

padding: 2rem;

border-radius: 15px;

box-shadow: 0 5px 20px rgba(0,0,0,0.1);

margin-bottom: 1.5rem;

}

.metric-container {

display: flex;

justify-content: space-around;

margin: 2rem 0;

}

.metric-box {

background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

color: white;

padding: 1.5rem;

border-radius: 10px;

text-align: center;
