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
     
    .position-underpriced { 
        background: #4facfe; 
    } 
     
    .position-competitive { 
        background: #43e97b; 
    } 
     
    .position-premium { 
        background: #fa709a; 
    } 
     
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

# Model Training Section 
st.markdown(""" 
<div style="background: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;"> 
    <h2 style="color: #667eea;">🤖 Step 1: Train Machine Learning Models</h2> 
</div> 
""", unsafe_allow_html=True) 

col1, col2 = st.columns([2, 1]) 

with col1: 
    st.markdown(""" 
    ### Model Training Pipeline 
     
    The system will train and compare three regression models: 
     
    1. **Linear Regression** (Baseline) 
       - Simple, interpretable 
       - Assumes linear relationships 
       - Fast training and prediction

    2. **Random Forest Regressor** 
       - Handles non-linear relationships 
       - Robust to outliers 
       - Feature importance analysis 
     
    3. **XGBoost Regressor** 
       - State-of-the-art performance 
       - Gradient boosting algorithm 
       - Best for complex patterns 
     
    The best model will be automatically selected based on test performance (MAE, RMSE, R²). 
    """) 

with col2: 
    st.markdown(""" 
    ### Training Settings 
    """) 
     
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05) 
     
    st.info(f""" 
    **Dataset Info:** 
    - Total samples: {len(df)} 
    - Training: {int(len(df) * (1-test_size))} 
    - Testing: {int(len(df) * test_size)} 
    """) 

if st.button("🚀 Train Models", type="primary"): 
    with st.spinner("Training models... This may take a few minutes."): 
        try: 
            # Initialize trainer 
            trainer = ModelTrainer() 
             
            # Prepare data 
            X_train, X_test, y_train, y_test = trainer.prepare_data(df, test_size=test_size) 
             
            # Train models 
            results = trainer.train_models(X_train, X_test, y_train, y_test) 
             
            # Save model 
            os.makedirs('models', exist_ok=True) 
            trainer.save_model('models/best_model.pkl')
            
            st.session_state.model_trained = True 
            st.session_state.model_results = results 
             
            st.success(f"✅ Model training completed! Best model: {trainer.best_model_name}") 
             
            # Display results 
            st.markdown("### 📊 Model Comparison") 
             
            comparison_data = [] 
            for model_name, result in results.items(): 
                comparison_data.append({ 
                    'Model': model_name, 
                    'Test MAE': f"${result['test']['mae']:.2f}", 
                    'Test RMSE': f"${result['test']['rmse']:.2f}", 
                    'Test R²': f"{result['test']['r2']:.4f}", 
                    'Train MAE': f"${result['train']['mae']:.2f}", 
                    'Train R²': f"{result['train']['r2']:.4f}" 
                }) 
             
            comparison_df = pd.DataFrame(comparison_data) 
            st.dataframe(comparison_df, use_container_width=True) 
             
            # Feature importance 
            st.markdown("### 🎯 Feature Importance") 
            importance_df = trainer.get_feature_importance() 
            st.dataframe(importance_df.head(10), use_container_width=True) 
             
        except Exception as e: 
            st.error(f"❌ Error training models: {str(e)}") 
            st.exception(e) 

# Price Prediction Section 
st.markdown(""" 
<div style="background: white; padding: 2rem; border-radius: 15px; margin: 2rem 0;"> 
    <h2 style="color: #667eea;">💡 Step 2: Get Price Recommendation</h2> 
</div> 
""", unsafe_allow_html=True) 

# Check if model exists 
model_exists = os.path.exists('models/best_model.pkl') 

if not model_exists: 
    st.warning("⚠️ No trained model found. Please train models first.") 
else: 
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True) 
     
    st.markdown("### Enter Product Details") 
     
    col1, col2, col3 = st.columns(3) 
     
    with col1: 
        category = st.selectbox("Product Category", df['category'].unique()) 
        brand = st.selectbox("Brand", df['brand'].unique()) 
        brand_tier = st.selectbox("Brand Tier", ['Budget', 'Mid-Range', 'Premium']) 
     
    with col2: 
        rating = st.slider("Expected Rating", 1.0, 5.0, 4.0, 0.1) 
        review_count = st.number_input("Expected Review Count", 0, 10000, 100) 
        platform = st.selectbox("Platform", df['platform'].unique()) 
     
    with col3: 
        # Calculate market averages from data 
        category_data = df[df['category'] == category] 
        market_avg = category_data['price'].mean() 
        market_median = category_data['price'].median() 
         
        st.metric("Market Avg Price", f"${market_avg:.2f}") 
        st.metric("Market Median Price", f"${market_median:.2f}") 
        st.metric("Competitor Count", len(category_data)) 
     
    st.markdown('</div>', unsafe_allow_html=True) 
     
    if st.button("🎯 Get Price Recommendation", type="primary"): 
        with st.spinner("Generating recommendation..."): 
            try: 
                # Load recommender 
                recommender = PriceRecommender('models/best_model.pkl') 
                 
                # Prepare features 
                brand_tier_map = {'Budget': 0, 'Mid-Range': 1, 'Premium': 2} 
                brand_encoded = df[df['brand'] == brand]['brand_encoded'].iloc[0] if len(df[df['brand'] == brand]) > 0 else 0 
                 
                # Calculate features 
                demand_score = rating * (review_count ** 0.5) 
                price_deviation = 0  # Will be calculated
                competitor_density = len(category_data) 
                 
                product_features = { 
                    'category': category, 
                    'brand': brand, 
                    'brand_tier': brand_tier, 
                    'brand_tier_encoded': brand_tier_map[brand_tier], 
                    'brand_encoded': brand_encoded, 
                    'rating': rating, 
                    'review_count': review_count, 
                    'platform': platform, 
                    'market_avg_price': market_avg, 
                    'market_median_price': market_median, 
                    'price_deviation': 0, 
                    'demand_score': demand_score, 
                    'brand_position_score': 50, 
                    'competitor_density': competitor_density, 
                    'price_rating_ratio': market_avg / rating, 
                    'review_popularity': 50 
                } 
                 
                # Add platform encoding 
                for p in df['platform'].unique(): 
                    product_features[f'platform_{p}'] = 1 if p == platform else 0 
                 
                # Get recommendation 
                recommendation = recommender.recommend_price(product_features, df) 
                 
                # Display recommendation 
                st.markdown(f""" 
                <div class="price-display"> 
                    <h2>Recommended Launch Price</h2> 
                    <div class="price-value">${recommendation['recommended_price']:.2f}</div> 
                    <div class="price-range"> 
                        Price Range: ${recommendation['price_range']['minimum']:.2f} - ${recommendation['price_range']['maximum']:.2f} 
                    </div> 
                    <div class="position-badge position-{recommendation['market_position']}"> 
                        {recommendation['positioning_label']} 
                    </div> 
                    <p style="margin-top: 1rem; opacity: 0.9;"> 
                        Confidence Score: {recommendation['confidence_score']:.0%} 
                    </p>
                </div> 
                """, unsafe_allow_html=True) 
                 
                # Strategy recommendations 
                st.markdown("### 📋 Pricing Strategy") 
                 
                strategy = recommendation['strategy'] 
                 
                col1, col2 = st.columns(2) 
                 
                with col1: 
                    st.markdown(f""" 
                    <div class="strategy-box"> 
                        <h4>🎯 Primary Strategy</h4> 
                        <p><strong>{strategy['primary_strategy']}</strong></p> 
                    </div> 
                    """, unsafe_allow_html=True) 
                     
                    st.markdown(""" 
                    <div class="strategy-box"> 
                        <h4>✅ Recommendations</h4> 
                    """, unsafe_allow_html=True) 
                    for rec in strategy['recommendations']: 
                        st.markdown(f"- {rec}") 
                    st.markdown("</div>", unsafe_allow_html=True) 
                 
                with col2: 
                    st.markdown(""" 
                    <div class="strategy-box"> 
                        <h4>⚠️ Risks</h4> 
                    """, unsafe_allow_html=True) 
                    for risk in strategy['risks']: 
                        st.markdown(f"- {risk}") 
                    st.markdown("</div>", unsafe_allow_html=True) 
                     
                    st.markdown(""" 
                    <div class="strategy-box"> 
                        <h4>💡 Opportunities</h4> 
                    """, unsafe_allow_html=True) 
                    for opp in strategy['opportunities']: 
                        st.markdown(f"- {opp}") 
                    st.markdown("</div>", unsafe_allow_html=True)

                # Competitive analysis 
                st.markdown("### 🔍 Competitive Analysis") 
                 
                similar_products = category_data[ 
                    (category_data['price'] >= recommendation['price_range']['minimum']) & 
                    (category_data['price'] <= recommendation['price_range']['maximum']) 
                ] 
                 
                col1, col2, col3 = st.columns(3) 
                 
                with col1: 
                    st.metric("Competitors in Price Range", len(similar_products)) 
                 
                with col2: 
                    st.metric("Your Price vs Market Avg",  
                             f"{((recommendation['recommended_price'] / market_avg - 1) * 100):.1f}%") 
                 
                with col3: 
                    percentile = (category_data['price'] < recommendation['recommended_price']).sum() / len(category_data) * 100 
                    st.metric("Price Percentile", f"{percentile:.0f}th") 
                 
            except Exception as e: 
                st.error(f"❌ Error generating recommendation: {str(e)}") 
                st.exception(e) 

# Additional Info 
st.markdown(""" 
<div style="background: white; padding: 2rem; border-radius: 15px; margin-top: 2rem;"> 
    <h3 style="color: #667eea;">📚 Understanding the Recommendation</h3> 
     
    <h4>Market Positioning:</h4> 
    <ul> 
        <li><strong>Budget/Value:</strong> Price < 85% of market average - Targets price-sensitive customers</li> 
        <li><strong>Competitive:</strong> Price within 85-115% of market average - Balanced market positioning</li> 
        <li><strong>Premium:</strong> Price > 115% of market average - Targets quality-focused customers</li> 
    </ul> 
     
    <h4>Price Range Explanation:</h4> 
    <p>The recommended range provides flexibility based on:</p>

    <ul> 
        <li>Market conditions and competition</li> 
        <li>Brand positioning and tier</li> 
        <li>Product features and quality</li> 
        <li>Launch timing and strategy</li> 
    </ul> 
     
    <h4>Confidence Score:</h4> 
    <p>Based on data quality, feature completeness, and market data availability. Higher scores indicate more reliable predictions.</p> 
</div> 
""", unsafe_allow_html=True)
