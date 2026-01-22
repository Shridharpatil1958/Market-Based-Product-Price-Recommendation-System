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
        min-width: 150px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .section-title {
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

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1 style="margin: 0;">📊 Live Market Data Collection</h1>
    <p style="margin-top: 1rem; opacity: 0.9;">Fetch and analyze real-time product pricing from multiple platforms</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'featured_data' not in st.session_state:
    st.session_state.featured_data = None

# Data Collection Section
st.markdown('<div class="section-title">🔍 Data Collection Settings</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    
    category = st.selectbox(
        "Select Product Category",
        ["smartphones", "laptops", "headphones", "tablets", "smartwatches"],
        help="Choose the product category to analyze"
    )
    
    max_results = st.slider(
        "Number of Products to Fetch",
        min_value=20,
        max_value=200,
        value=100,
        step=10,
        help="More products = better model accuracy but slower processing"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📝 Data Sources
    
    This system fetches data from:
    - **Amazon** Product Advertising API
    - **eBay** Browse API
    - **Walmart** Marketplace
    - **BestBuy** Product Catalog
    - **Target** Retail API
    
    *Note: Demo version uses synthetic but realistic data*
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Fetch Data Button
if st.button("🚀 Fetch Market Data", type="primary"):
    with st.spinner("Fetching live market data..."):
        try:
            # Initialize API client
            api_client = ProductPriceAPI()
            
            # Fetch data
            raw_df = api_client.fetch_product_data(category, max_results)
            st.session_state.raw_data = raw_df
            
            # Get summary
            summary = api_client.get_category_summary(raw_df)
            
            st.success(f"✅ Successfully fetched {len(raw_df)} products!")
            
            # Display summary metrics
            st.markdown('<div class="section-title">📈 Data Summary</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{summary['total_products']}</div>
                    <div class="metric-label">Total Products</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{summary['brands']}</div>
                    <div class="metric-label">Brands</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">${summary['avg_price']:.2f}</div>
                    <div class="metric-label">Avg Price</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{summary['avg_rating']:.1f}★</div>
                    <div class="metric-label">Avg Rating</div>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")

# Display Raw Data
if st.session_state.raw_data is not None:
    st.markdown('<div class="section-title">📋 Raw Market Data</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    
    # Display data info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(st.session_state.raw_data, use_container_width=True, height=400)
    
    with col2:
        st.markdown("### Data Quality Issues")
        missing = st.session_state.raw_data.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing': missing.values
        })
        missing_df = missing_df[missing_df['Missing'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values detected!")
        
        duplicates = st.session_state.raw_data.duplicated().sum()
        st.metric("Duplicate Rows", duplicates)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Cleaning Section
    st.markdown('<div class="section-title">🧹 Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)
    
    if st.button("🔧 Clean and Process Data", type="primary"):
        with st.spinner("Cleaning data..."):
            try:
                # Clean data
                cleaner = DataCleaner()
                cleaned_df = cleaner.clean_data(st.session_state.raw_data.copy())
                st.session_state.cleaned_data = cleaned_df
                
                # Feature engineering
                engineer = FeatureEngineer()
                featured_df = engineer.create_features(cleaned_df)
                st.session_state.featured_data = featured_df
                
                # Get cleaning summary
                summary = cleaner.get_cleaning_summary(st.session_state.raw_data, cleaned_df)
                
                st.success("✅ Data cleaning completed!")
                
                # Display cleaning summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rows Before", summary['rows_before'])
                    st.metric("Rows After", summary['rows_after'])
                
                with col2:
                    st.metric("Rows Removed", summary['rows_removed'])
                    st.metric("Removal %", f"{summary['removal_percentage']:.1f}%")
                
                with col3:
                    st.metric("Missing Before", summary['missing_values_before'])
                    st.metric("Missing After", summary['missing_values_after'])
                
                # Save data
                os.makedirs('data/processed', exist_ok=True)
                featured_df.to_csv('data/processed/featured_data.csv', index=False)
                st.info("💾 Processed data saved to data/processed/featured_data.csv")
                
            except Exception as e:
                st.error(f"❌ Error cleaning data: {str(e)}")

# Display Cleaned Data
if st.session_state.featured_data is not None:
    st.markdown('<div class="section-title">✨ Cleaned & Featured Data</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "🔢 Statistics", "📈 Feature Summary"])
    
    with tab1:
        st.dataframe(st.session_state.featured_data, use_container_width=True, height=400)
        
        # Download button
        csv = st.session_state.featured_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data",
            data=csv,
            file_name="processed_market_data.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.write("### Descriptive Statistics")
        st.dataframe(st.session_state.featured_data.describe(), use_container_width=True)
    
    with tab3:
        engineer = FeatureEngineer()
        feature_summary = engineer.get_feature_summary(st.session_state.featured_data)
        st.dataframe(feature_summary, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Instructions
st.markdown('<div class="section-title">💡 Next Steps</div>', unsafe_allow_html=True)

st.markdown("""
<div class="data-card">
    <h3 style="color: #667eea;">What to do next?</h3>
    
    <ol style="line-height: 2;">
        <li>Review the fetched market data and cleaning results</li>
        <li>Go to <strong>Analytics Dashboard</strong> to visualize market trends</li>
        <li>Use <strong>Price Recommendation</strong> to train models and get predictions</li>
    </ol>
    
    <p style="color: #666; margin-top: 1rem;">
        The cleaned and featured data is ready for model training!
    </p>
</div>
""", unsafe_allow_html=True)