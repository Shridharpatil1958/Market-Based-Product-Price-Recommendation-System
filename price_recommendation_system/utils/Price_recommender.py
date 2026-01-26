import streamlit as st
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import your pipeline modules
from api_client import ProductPriceAPI
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer

class PriceRecommender:
    """
    Complete product price recommendation system.
    Integrates data collection, cleaning, feature engineering, and ML prediction.
    
    Production-ready Streamlit application for product price recommendations.
    """

    def __init__(self):
        self.api_client = ProductPriceAPI()
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.loaded_model = None

    def load_trained_model(self, model_path: str = "best_price_model.pkl"):
        """
        Load pre-trained model for predictions.
        
        Args:
            model_path: Path to saved model file
        """
        try:
            self.loaded_model = self.model_trainer.load_model(model_path)
            st.success(f"✅ Model loaded successfully: {model_path}")
            return True
        except FileNotFoundError:
            st.warning("⚠ Model file not found. Please train a model first.")
            return False

    def run_complete_pipeline(self, category: str, num_products: int = 200) -> pd.DataFrame:
        """
        Execute complete end-to-end pipeline:
        1. Data collection → 2. Cleaning → 3. Feature engineering → 4. Price prediction
        
        Args:
            category: Product category to analyze
            num_products: Number of products to fetch
            
        Returns:
            DataFrame with predicted prices and recommendations
        """
        with st.spinner(f"🔄 Running complete pipeline for {category}..."):
            
            # Step 1: Data Collection
            st.info("📊 Step 1: Fetching product data...")
            raw_data = self.api_client.fetch_product_data(category, num_products)
            
            # Step 2: Data Cleaning
            st.info("🧹 Step 2: Cleaning data...")
            cleaned_data = self.data_cleaner.clean_data(raw_data)
            
            # Step 3: Feature Engineering
            st.info("🔧 Step 3: Engineering features...")
            engineered_data = self.feature_engineer.engineer_features(cleaned_data)
            
            # Step 4: Price Prediction
            if self.loaded_model is not None:
                st.info("🤖 Step 4: Predicting optimal prices...")
                engineered_data = self._predict_optimal_prices(engineered_data)
            else:
                st.warning("⚠ No trained model found. Skipping predictions.")
            
            return engineered_data

    def _predict_optimal_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict optimal pricing using trained model.
        """
        # Prepare features for prediction
        X = df.drop(columns=['price', 'original_price'])
        
        # Convert categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = pd.factorize(X[col])[0]
        
        # Make predictions
        predicted_prices = self.loaded_model.predict(X)
        df['predicted_price'] = predicted_prices
        
        # Calculate recommendations
        df['recommendation'] = self._generate_recommendations(df)
        df['price_gap'] = df['predicted_price'] - df['price']
        
        return df

    def _generate_recommendations(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate actionable price recommendations.
        """
        conditions = [
            (df['price_gap'] < -10),  # Price too high
            (df['price_gap'] > 10),   # Price too low
            (abs(df['price_gap']) <= 10)  # Good price
        ]
        
        choices = ['📈 Increase Price', '📉 Decrease Price', '✅ Optimal Price']
        
        return np.select(conditions, choices, default='⚠ Review Pricing')

    def get_price_insights(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive price insights and statistics.
        """
        insights = {
            'avg_current_price': df['price'].mean(),
            'avg_predicted_price': df['predicted_price'].mean(),
            'avg_price_gap': df['price_gap'].mean(),
            'recommend_increase': len(df[df['recommendation'] == '📈 Increase Price']),
            'recommend_decrease': len(df[df['recommendation'] == '📉 Decrease Price']),
            'optimal_price': len(df[df['recommendation'] == '✅ Optimal Price']),
            'price_accuracy': np.mean(abs(df['price_gap']) < 5) * 100
        }
        
        return insights

def main():
    """
    Streamlit application main function.
    Production-ready price recommendation dashboard.
    """
    st.set_page_config(
        page_title="🚀 Product Price Recommender",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("💰 Product Price Recommender")
    st.markdown("---")
    st.markdown("""
    **Intelligent pricing recommendations powered by machine learning.**
    
    *End-to-end ML pipeline: Data Collection → Cleaning → Feature Engineering → Price Prediction*
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Pipeline Configuration")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model File Path", 
        value="best_price_model.pkl",
        help="Path to your trained model file"
    )
    
    # Pipeline parameters
    category = st.sidebar.selectbox(
        "Product Category",
        options=['smartphones', 'laptops', 'headphones', 'tablets', 'smartwatches']
    )
    
    num_products = st.sidebar.slider(
        "Number of Products to Analyze",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )
    
    # Initialize recommender
    recommender = PriceRecommender()
    
    # Load model
    if st.sidebar.button("📂 Load Trained Model", type="primary"):
        if recommender.load_trained_model(model_path):
            st.sidebar.success("✅ Model loaded!")
        else:
            st.sidebar.error("❌ Failed to load model")
    
    # Main pipeline execution
    if st.button("🚀 Run Complete Pipeline", type="primary", use_container_width=True):
        
        # Run pipeline
        results_df = recommender.run_complete_pipeline(category, num_products)
        
        # Store results in session state
        st.session_state.results_df = results_df
        st.session_state.category = category
        
        st.success("✅ Pipeline completed successfully!")
    
    # Results visualization (if results exist)
    if 'results_df' in st.session_state:
        df = st.session_state.results_df
        
        # Key Insights
        st.markdown("## 📊 Key Insights")
        col1, col2, col3, col4 = st.columns(4)
        
        insights = recommender.get_price_insights(df)
        
        with col1:
            st.metric("Avg Current Price", f"${insights['avg_current_price']:.2f}")
        with col2:
            st.metric("Avg Predicted Price", f"${insights['avg_predicted_price']:.2f}")
        with col3:
            st.metric("Price Accuracy", f"{insights['price_accuracy']:.1f}%")
        with col4:
            st.metric("Net Price Adjustment", f"${insights['avg_price_gap']:.2f}")
        
        # Recommendations Summary
        st.markdown("## 🎯 Price Recommendations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Increase Price", insights['recommend_increase'])
        with col2:
            st.metric("Decrease Price", insights['recommend_decrease'])
        with col3:
            st.metric("Optimal Price", insights['optimal_price'])
        
        # Interactive Tables
        st.markdown("---")
        
        # Top price opportunities
        st.markdown("## 🔥 Top Price Opportunities")
        opportunities = df.nlargest(10, 'price_gap').copy()
        opportunities['price_gap'] = opportunities['price_gap'].round(2)
        opportunities['predicted_price'] = opportunities['predicted_price'].round(2)
        st.dataframe(opportunities[['product_name', 'brand', 'price', 'predicted_price', 
                                   'price_gap', 'recommendation']].style.format({
                'price': '${:.2f}',
                'predicted_price': '${:.2f}',
                'price_gap': '${:.2f}'
            }), use_container_width=True)
        
        # Price distribution
        st.markdown("## 📈 Price Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current vs Predicted Prices")
            fig_data = {
                'Current Price': df['price'],
                'Predicted Price': df['predicted_price']
            }
            df_plot = pd.DataFrame(fig_data)
            st.bar_chart(df_plot)
        
        with col2:
            st.subheader("Recommendations Distribution")
            rec_counts = df['recommendation'].value_counts()
            st.bar_chart(rec_counts)
        
        # Detailed results
        st.markdown("## 📋 Detailed Results")
        st.dataframe(df[['product_name', 'brand', 'platform', 'price', 'predicted_price',
                        'price_gap', 'recommendation', 'rating', 'review_count']].style.format({
                'price': '${:.2f}',
                'predicted_price': '${:.2f}',
                'price_gap': '${:.2f}'
            }), use_container_width=True)
    
    else:
        st.info("👆 Click 'Run Complete Pipeline' to analyze pricing data")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Built with ❤️ using Streamlit + Complete ML Pipeline**
    
    *Data Analyst Portfolio Project - End-to-End Price Recommendation System*
    """)

if __name__ == "__main__":
    main()
