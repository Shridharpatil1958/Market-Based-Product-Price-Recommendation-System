import streamlit as st 
import pandas as pd 
import sys 
import os 
import joblib 
 
# Add parent directory to path 
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
 
from utils.model_trainer import ModelTrainer 
from utils.price_recommender import PriceRecommender 
 
# Page config 
st.set_page_config(page_title="Price Recommendation", page_icon="💰", layout="wide") 
 
# Custom CSS 
st.markdown(""" 
<style> 
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 2rem; 
    } 
     
    .prediction-card { 
        background: white; 
        padding: 2rem; 
        border-radius: 15px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
        margin-bottom: 1.5rem; 
    } 
     
    .price-display { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white; 
        padding: 2rem; 
        border-radius: 15px; 
        text-align: center; 
        margin: 2rem 0; 
    } 
     
    .price-value { 
        font-size: 3.5rem; 
        font-weight: 700; 
        margin: 1rem 0; 
    } 
     
    .price-range { 
        font-size: 1.2rem; 
        opacity: 0.9; 
    } 
     
    .position-badge { 
        display: inline-block; 
        padding: 0.5rem 1.5rem; 
        border-radius: 20px; 
        font-weight: 600; 
        margin-top: 1rem; 
    } 
     
    .position-underpriced { background: #4facfe; } 
    .position-competitive { background: #43e97b; } 
    .position-premium { background: #fa709a; } 
     
    .strategy-box { 
        background: #f8f9fa; 
        padding: 1.5rem; 
        border-radius: 10px; 
        border-left: 4px solid #667eea; 
        margin: 1rem 0; 
    } 
</style> 
""", unsafe_allow_html=True) 
 
# Header 
st.markdown(""" 
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  
            padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1 style="margin: 0;">💰 Price Recommendation Engine</h1> 
    <p style="margin-top: 1rem; opacity: 0.9;">AI-Powered Optimal Pricing for New Product Launch</p> 
</div> 
""", unsafe_allow_html=True) 
 
# Initialize session state 
if 'model_trained' not in st.session_state: 
    st.session_state.model_trained = False 
if 'model_results' not in st.session_state: 
    st.session_state.model_results = None 
 
# Check if data is available 
data_available = os.path.exists('data/processed/featured_data.csv') 
if not data_available: 
    st.warning("⚠️ No processed data found. Please fetch and process data in the 'Live Market Data' page first.") 
    st.stop() 
 
# Load data 
df = pd.read_csv('data/processed/featured_data.csv')
