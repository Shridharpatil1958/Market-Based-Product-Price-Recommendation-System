# Market-Based Product Price Recommendation System - Development Plan

## Project Structure
```
/workspace
├── app.py                          # Main Streamlit application with multi-page navigation
├── requirements.txt                # Python dependencies
├── pages/
│   ├── 1_Live_Market_Data.py      # API data fetching and display
│   ├── 2_Price_Recommendation.py  # Price prediction interface
│   └── 3_Analytics_Dashboard.py   # EDA visualizations
├── utils/
│   ├── api_client.py              # API integration functions
│   ├── data_cleaner.py            # Data cleaning and preprocessing
│   ├── feature_engineer.py        # Feature engineering functions
│   ├── model_trainer.py           # ML model training and evaluation
│   └── price_recommender.py       # Price recommendation logic
├── data/
│   ├── raw/                       # Raw API data
│   └── processed/                 # Cleaned data
├── models/
│   └── best_model.pkl             # Saved trained model
└── assets/
    └── styles.css                 # Custom CSS styles
```

## Implementation Tasks

### 1. Project Setup & Dependencies
- Create requirements.txt with all necessary packages
- Initialize folder structure
- Create main app.py with custom HTML styling

### 2. API Integration Module (utils/api_client.py)
- Implement RapidAPI product price fetching
- Support multiple product categories
- Handle API errors and rate limits
- Return structured data (name, category, brand, price, rating, reviews, platform)

### 3. Data Cleaning Pipeline (utils/data_cleaner.py)
- Handle missing values (forward fill, median imputation)
- Remove duplicate products based on name+brand
- Convert price strings to numeric
- Normalize prices across platforms
- Outlier detection: IQR method (Q1-1.5*IQR, Q3+1.5*IQR)
- Outlier detection: Z-score method (|z| > 3)
- Categorical encoding (Label Encoding for brand tier, One-Hot for category)
- Document each cleaning step with business justification

### 4. EDA Module (pages/3_Analytics_Dashboard.py)
- Price distribution histogram with KDE
- Box plots for price across platforms
- Average price per category bar chart
- Scatter plot: Price vs Rating
- Line plot: Review count vs Price trends
- Heatmap: Correlation matrix
- Competitive price bands visualization

### 5. Feature Engineering (utils/feature_engineer.py)
- Market Average Price: mean price in category
- Market Median Price: median price in category
- Price Deviation: (price - market_avg) / market_avg
- Demand Score: rating × log(review_count + 1)
- Brand Position Score: percentile rank of brand avg price
- Competitor Density: count of products in ±10% price range
- Business explanations for each feature

### 6. ML Pipeline (utils/model_trainer.py)
- Train-test split (80-20)
- Feature scaling using StandardScaler
- Models:
  - Linear Regression (baseline)
  - Random Forest Regressor (n_estimators tuning)
  - XGBoost Regressor (learning_rate, max_depth tuning)
- GridSearchCV for hyperparameter tuning
- Evaluation: MAE, RMSE, R² Score
- Model comparison table
- Save best model

### 7. Price Recommendation Engine (utils/price_recommender.py)
- Load trained model
- Predict optimal price
- Calculate price range: [predicted * 0.9, predicted * 1.1]
- Market positioning logic:
  - Underpriced: predicted < market_avg * 0.85
  - Competitive: 0.85 ≤ predicted/market_avg ≤ 1.15
  - Premium: predicted > market_avg * 1.15
- Pricing strategy recommendations

### 8. Streamlit UI (HTML/CSS Only)
- Home page: Project overview with custom HTML cards
- Live Market Data: API fetch interface with data table
- Price Recommendation: Input form with prediction output
- Analytics Dashboard: Interactive charts
- Custom CSS: Modern gradient design, cards, buttons
- No React, pure HTML with st.markdown(unsafe_allow_html=True)

## Development Order
1. Setup (requirements.txt, folder structure)
2. API client implementation
3. Data cleaning utilities
4. Feature engineering functions
5. Model training pipeline
6. Price recommendation logic
7. Streamlit pages with custom HTML/CSS
8. Testing and integration