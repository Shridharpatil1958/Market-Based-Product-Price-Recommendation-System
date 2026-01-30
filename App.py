import streamlit as st 
import pandas as pd 
import sys 
import os 

# Add utils to path 
sys.path.append(os.path.dirname(__file__)) 

# Page configuration 
st.set_page_config( 
    page_title="Market-Based Price Recommendation System", 
    page_icon="💰", 
    layout="wide", 
    initial_sidebar_state="expanded" 
) 

# Custom CSS with modern design 
st.markdown(""" 
<style> 
    /* Import Google Fonts */ 
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap'); 
     
    /* Global Styles */ 
    * { 
        font-family: 'Inter', sans-serif; 
    } 
     
    /* Main container */ 
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 2rem; 
    } 
     
    /* Hero section */ 
    .hero-section { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 3rem 2rem; 
        border-radius: 20px; 
        color: white; 
        text-align: center; 
        margin-bottom: 2rem; 
        box-shadow: 0 10px 40px rgba(0,0,0,0.2); 
    } 
     
    .hero-title { 
        font-size: 3rem; 
        font-weight: 700; 
        margin-bottom: 1rem; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3); 
    } 
     
    .hero-subtitle { 
        font-size: 1.3rem; 
        font-weight: 300; 
        opacity: 0.95; 
    } 
     
    /* Feature cards */ 
    .feature-card { 
        background: white; 
        padding: 2rem; 
        border-radius: 15px; 
        box-shadow: 0 5px 20px rgba(0,0,0,0.1); 
        margin-bottom: 1.5rem; 
        transition: transform 0.3s ease, box-shadow 0.3s ease; 
    } 
     
    .feature-card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 10px 30px rgba(0,0,0,0.15); 
    } 
     
    .feature-icon { 
        font-size: 2.5rem; 
        margin-bottom: 1rem; 
    } 
     
    .feature-title { 
        font-size: 1.5rem; 
        font-weight: 600; 
        color: #667eea; 
        margin-bottom: 0.5rem; 
    } 
     
    .feature-description { 
        color: #666; 
        line-height: 1.6; 
    } 
     
    /* Info boxes */ 
    .info-box { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
        color: white; 
        padding: 1.5rem; 
        border-radius: 10px; 
        margin: 1rem 0; 
    } 
     
    .success-box { 
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
        color: white; 
        padding: 1.5rem; 
        border-radius: 10px; 
        margin: 1rem 0; 
    } 
     
    .warning-box { 
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
        color: white; 
        padding: 1.5rem; 
        border-radius: 10px; 
        margin: 1rem 0; 
    } 
     
    /* Buttons */ 
    .stButton>button { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white; 
        border: none; 
        padding: 0.75rem 2rem; 
        border-radius: 10px; 
        font-weight: 600; 
        transition: all 0.3s ease; 
    } 
     
    .stButton>button:hover { 
        transform: scale(1.05); 
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); 
    } 
     
    /* Sidebar */ 
    .css-1d391kg { 
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); 
    } 
     
    /* Metrics */ 
    .metric-card { 
        background: white; 
        padding: 1.5rem; 
        border-radius: 10px; 
        text-align: center; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.1); 
    } 
     
    .metric-value { 
        font-size: 2rem; 
        font-weight: 700; 
        color: #667eea; 
    } 
     
    .metric-label { 
        color: #666; 
        font-size: 0.9rem; 
        margin-top: 0.5rem; 
    } 
     
    /* Tables */ 
    .dataframe { 
        border-radius: 10px; 
        overflow: hidden; 
    } 
     
    /* Section headers */ 
    .section-header { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white; 
        padding: 1rem 1.5rem; 
        border-radius: 10px; 
        margin: 2rem 0 1rem 0; 
        font-size: 1.5rem; 
        font-weight: 600; 
    } 
</style> 
""", unsafe_allow_html=True) 

# Hero Section 
st.markdown(""" 
<div class="hero-section"> 
    <div class="hero-title">💰 Market-Based Price Recommendation System</div> 
    <div class="hero-subtitle"> 
        AI-Powered Optimal Pricing for New Product Launches 
    </div> 
</div> 
""", unsafe_allow_html=True) 

# Introduction 
st.markdown(""" 
<div class="feature-card"> 
    <div class="feature-icon">🎯</div> 
    <div class="feature-title">Welcome to the Price Recommendation System</div> 
    <div class="feature-description"> 
        This end-to-end Data Science application helps you determine the optimal launch price  
        for new products by analyzing live market data, training machine learning models,  
        and providing strategic pricing recommendations. 
    </div> 
</div> 
""", unsafe_allow_html=True) 

# Features Section 
col1, col2, col3 = st.columns(3) 

with col1: 
    st.markdown(""" 
    <div class="feature-card"> 
        <div class="feature-icon">📊</div> 
        <div class="feature-title">Live Market Data</div> 
        <div class="feature-description"> 
            Fetch real-time product pricing from multiple platforms including Amazon, eBay,  
            and more. Analyze competitor pricing across categories. 
        </div> 
    </div> 
    """, unsafe_allow_html=True) 

with col2: 
    st.markdown(""" 
    <div class="feature-card"> 
        <div class="feature-icon">🤖</div> 
        <div class="feature-title">ML-Powered Predictions</div> 
        <div class="feature-description"> 
            Advanced machine learning models (Linear Regression, Random Forest, XGBoost)  
            trained on market data to predict optimal prices. 
        </div> 
    </div> 
    """, unsafe_allow_html=True) 

with col3: 
    st.markdown(""" 
    <div class="feature-card"> 
        <div class="feature-icon">💡</div> 
        <div class="feature-title">Strategic Insights</div> 
        <div class="feature-description"> 
            Get market positioning analysis, pricing strategy recommendations,  
            and competitive intelligence for your product launch. 
        </div> 
    </div> 
    """, unsafe_allow_html=True) 

# How It Works 
st.markdown('<div class="section-header">🔄 How It Works</div>', unsafe_allow_html=True) 

st.markdown(""" 
<div class="feature-card"> 
    <h3 style="color: #667eea;">End-to-End Data Science Pipeline</h3> 
     
    <ol style="line-height: 2; color: #333;"> 
        <li><strong>Data Collection:</strong> Fetch live product pricing data from multiple APIs</li> 
        <li><strong>Data Cleaning:</strong> Handle missing values, remove duplicates, detect outliers</li> 
        <li><strong>Feature Engineering:</strong> Create advanced features like demand score, competitor density</li> 
        <li><strong>Model Training:</strong> Train and compare multiple ML models (Linear Regression, Random Forest, XGBoost)</li> 
        <li><strong>Price Prediction:</strong> Generate optimal price with confidence intervals</li> 
        <li><strong>Strategic Analysis:</strong> Provide market positioning and pricing strategy recommendations</li> 
    </ol> 
</div> 
""", unsafe_allow_html=True) 

# Key Features 
st.markdown('<div class="section-header">⭐ Key Features</div>', unsafe_allow_html=True) 

col1, col2 = st.columns(2) 

with col1: 
    st.markdown(""" 
    <div class="info-box"> 
        <h4>📈 Advanced Analytics</h4> 
        <ul> 
            <li>Price distribution analysis</li> 
            <li>Competitive price bands</li> 
            <li>Rating vs Price correlation</li> 
            <li>Market trend visualization</li> 
        </ul> 
    </div> 
    """, unsafe_allow_html=True) 
     
    st.markdown(""" 
    <div class="success-box"> 
        <h4>🎯 Smart Recommendations</h4> 
        <ul> 
            <li>Optimal launch price</li> 
            <li>Price range (min-max)</li> 
            <li>Market positioning label</li> 
            <li>Pricing strategy insights</li> 
        </ul> 
    </div> 
    """, unsafe_allow_html=True) 

with col2: 
    st.markdown(""" 
    <div class="warning-box"> 
        <h4>🔬 Data Science Best Practices</h4> 
        <ul> 
            <li>Comprehensive data cleaning</li> 
            <li>Feature engineering with business logic</li> 
            <li>Model comparison & selection</li> 
            <li>Hyperparameter tuning</li> 
        </ul> 
    </div> 
    """, unsafe_allow_html=True) 
     
    st.markdown(""" 
    <div class="info-box"> 
        <h4>💼 Business Value</h4> 
        <ul> 
            <li>Reduce pricing guesswork</li> 
            <li>Maximize profit margins</li> 
            <li>Competitive market positioning</li> 
            <li>Data-driven decision making</li> 
        </ul> 
    </div> 
    """, unsafe_allow_html=True) 

# Navigation Guide 
st.markdown('<div class="section-header">🗺️ Navigation Guide</div>', unsafe_allow_html=True) 

st.markdown(""" 
<div class="feature-card"> 
    <h3 style="color: #667eea;">Explore the Application</h3> 
     
    <p style="color: #666; line-height: 1.8;"> 
        Use the sidebar to navigate through different sections: 
    </p> 
     
    <div style="margin: 1.5rem 0;"> 
        <p><strong style="color: #667eea;">📊 Live Market Data:</strong> Fetch and explore real-time product pricing from multiple platforms</p> 
        <p><strong style="color: #667eea;">💰 Price Recommendation:</strong> Get AI-powered price predictions for your new product</p> 
        <p><strong style="color: #667eea;">📈 Analytics Dashboard:</strong> View comprehensive market analysis and visualizations</p> 
    </div> 
</div> 
""", unsafe_allow_html=True) 

# Footer 
st.markdown(""" 
<div style="text-align: center; margin-top: 3rem; padding: 2rem; background: white; border-radius: 10px;">
    <p style="color: #666; font-size: 0.9rem;"> 
        Built with ❤️ using Streamlit, Scikit-learn, XGBoost, and Python 
    </p> 
    <p style="color: #999; font-size: 0.8rem;"> 
        © 2024 Market-Based Price Recommendation System | Data Science Application 
    </p> 
</div> 
""", unsafe_allow_html=True)
