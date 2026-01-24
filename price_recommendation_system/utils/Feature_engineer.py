import pandas as pd 
import numpy as np 
from typing import List, Tuple, Dict 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.decomposition import PCA 
import warnings 
warnings.filterwarnings('ignore') 

class FeatureEngineer: 
    """ 
    Advanced feature engineering for product pricing prediction. 
    Creates new features that capture market dynamics and product characteristics. 
    """ 
     
    def __init__(self): 
        self.feature_history = {} 
        self.scaler = StandardScaler() 
        self.pca = PCA() 
     
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Execute complete feature engineering pipeline. 
         
        Args: 
            df: Cleaned product DataFrame 
             
        Returns: 
            DataFrame with engineered features 
        """ 
        df_engineered = df.copy() 
         
        print("=" * 60) 
        print("FEATURE ENGINEERING PIPELINE") 
        print("=" * 60) 
         
        # Step 1: Create rating-based features 
        df_engineered = self._engineer_rating_features(df_engineered) 
         
        # Step 2: Create price-based features 
        df_engineered = self._engineer_price_features(df_engineered) 
         
        # Step 3: Create popularity features 
        df_engineered = self._engineer_popularity_features(df_engineered) 
         
        # Step 4: Create interaction features 
        df_engineered = self._engineer_interaction_features(df_engineered) 
         
        # Step 5: Create market-based features 
        df_engineered = self._engineer_market_features(df_engineered) 
         
        print("
" + "=" * 60) 
        print("FEATURE ENGINEERING COMPLETE") 
        print("=" * 60) 
         
        return df_engineered 
     
    def _engineer_rating_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Create features based on product ratings. 
         
        Why: Rating indicates customer satisfaction and product quality. 
        Features: 
        - rating_squared: Emphasizes high-rated products 
        - rating_category: Categorical buckets (Poor, Average, Good, Excellent) 
        - rating_deviation: How much product deviates from avg rating 
        """ 
        print("
1. ENGINEERING RATING-BASED FEATURES") 
        print("-" * 60) 
         
        # Feature 1: Rating squared (emphasize differences) 
        df['rating_squared'] = df['rating'] ** 2 
        print(f"✓ rating_squared: Emphasizes high-quality products") 
         
        # Feature 2: Rating categories 
        def categorize_rating(rating): 
            if rating < 2.0: 
                return 'Poor' 
            elif rating < 3.0: 
                return 'Average' 
            elif rating < 4.0: 
                return 'Good' 
            else: 
                return 'Excellent' 
         
        df['rating_category'] = df['rating'].apply(categorize_rating) 
        print(f"✓ rating_category: Categorical rating buckets") 
         
        # Feature 3: Rating deviation from mean 
        mean_rating = df['rating'].mean() 
        df['rating_deviation'] = df['rating'] - mean_rating 
        print(f"✓ rating_deviation: Difference from mean rating ({mean_rating:.2f})") 
         
        return df 
     
    def _engineer_price_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Create features based on product prices. 
         
        Why: Price is critical for market positioning and prediction. 
        Features: 
        - log_price: Handles skewed price distributions 
        - price_category: Low/Mid/High price brackets 
        - price_percentile: Product's position in price distribution 
        """ 
        print("
2. ENGINEERING PRICE-BASED FEATURES") 
        print("-" * 60) 
         
        # Feature 1: Log price (handle skewed distributions) 
        df['log_price'] = np.log1p(df['price']) 
        print(f"✓ log_price: Log transformation handles skewed distributions") 
         
        # Feature 2: Price categories 
        price_q33 = df['price'].quantile(0.33) 
        price_q67 = df['price'].quantile(0.67) 
         
        def categorize_price(price): 
            if price < price_q33: 
                return 'Budget' 
            elif price < price_q67: 
                return 'Mid-Range' 
            else: 
                return 'Premium' 
         
        df['price_category'] = df['price'].apply(categorize_price) 
        print(f"✓ price_category: Budget/Mid-Range/Premium brackets") 
         
        # Feature 3: Price percentile 
        df['price_percentile'] = df['price'].rank(pct=True) 
        print(f"✓ price_percentile: Percentile position in price distribution") 
         
        return df 
     
    def _engineer_popularity_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Create features based on review counts and engagement. 
         
        Why: More reviews = higher visibility and social proof. 
        Features: 
        - log_review_count: Log transformation of reviews 
        - review_per_rating: Reviews normalized by rating 
        - popularity_score: Composite popularity metric 
        """ 
        print("
3. ENGINEERING POPULARITY-BASED FEATURES") 
        print("-" * 60) 
         
        # Feature 1: Log review count 
        df['log_review_count'] = np.log1p(df['review_count']) 
        print(f"✓ log_review_count: Log transformation of review volume") 
         
        # Feature 2: Review engagement ratio 
        df['review_per_rating'] = df['review_count'] / (df['rating'] + 1) 
        print(f"✓ review_per_rating: Review volume relative to rating") 
         
        # Feature 3: Popularity score (composite) 
        # Normalize components 0-1 
        norm_reviews = (df['review_count'] - df['review_count'].min()) / \
                       (df['review_count'].max() - df['review_count'].min() + 1) 
        norm_rating = (df['rating'] - df['rating'].min()) / \
                      (df['rating'].max() - df['rating'].min() + 1) 
         
        df['popularity_score'] = (0.6 * norm_reviews + 0.4 * norm_rating) 
        print(f"✓ popularity_score: Weighted composite metric (60% reviews, 40% rating)") 
         
        return df 
     
    def _engineer_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Create interaction features between price and ratings. 
         
        Why: Price-quality relationship impacts perceived value. 
        Features: 
        - price_rating_ratio: Value proposition metric 
        - quality_adjusted_price: Ratio of price to rating 
        - value_score: Combined perceived value 
        """ 
        print("
4. ENGINEERING INTERACTION FEATURES") 
        print("-" * 60) 
         
        # Feature 1: Price-to-rating ratio 
        df['price_rating_ratio'] = df['price'] / (df['rating'] + 1) 
        print(f"✓ price_rating_ratio: Price per unit quality") 
         
        # Feature 2: Quality-adjusted price 
        df['quality_adjusted_price'] = df['price'] * (df['rating'] / 5.0) 
        print(f"✓ quality_adjusted_price: Price weighted by quality") 
         
        # Feature 3: Value score 
        norm_rating = (df['rating'] - df['rating'].min()) / \
                      (df['rating'].max() - df['rating'].min() + 1) 
        norm_price = (df['price'] - df['price'].min()) / \
                     (df['price'].max() - df['price'].min() + 1) 
         
        df['value_score'] = (norm_rating / (norm_price + 0.1)) 
        print(f"✓ value_score: Quality-to-price value proposition") 
         
        return df 
     
    def _engineer_market_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Create features based on market context and trends. 
         
        Why: Market dynamics affect pricing and competitiveness. 
        Features: 
        - brand_avg_price: Average price per brand 
        - platform_avg_price: Average price per platform 
        - price_vs_brand_avg: How product compares to its brand 
        - price_vs_platform_avg: How product compares within platform 
        """ 
        print("
5. ENGINEERING MARKET-BASED FEATURES") 
        print("-" * 60) 
         
        # Feature 1: Brand average price 
        brand_avg = df.groupby('brand')['price'].transform('mean') 
        df['brand_avg_price'] = brand_avg 
        print(f"✓ brand_avg_price: Average price per brand") 
         
        # Feature 2: Platform average price 
        platform_avg = df.groupby('platform')['price'].transform('mean') 
        df['platform_avg_price'] = platform_avg 
        print(f"✓ platform_avg_price: Average price per platform") 
         
        # Feature 3: Price vs brand average 
        df['price_vs_brand'] = df['price'] - df['brand_avg_price'] 
        print(f"✓ price_vs_brand: Price deviation from brand average") 
         
        # Feature 4: Price vs platform average 
        df['price_vs_platform'] = df['price'] - df['platform_avg_price'] 
        print(f"✓ price_vs_platform: Price deviation from platform average") 
         
        return df 
     
    def scale_features(self, df: pd.DataFrame, 
                      features_to_scale: List[str]) -> Tuple[pd.DataFrame, StandardScaler]: 
        """ 
        Standardize numeric features using StandardScaler. 
         
        Why: ML algorithms perform better with normalized features. 
         
        Args: 
            df: DataFrame with engineered features 
            features_to_scale: List of column names to scale 
             
        Returns: 
            Tuple of (scaled DataFrame, fitted scaler object) 
        """ 
        print("
" + "=" * 60) 
        print("FEATURE SCALING") 
        print("=" * 60) 
         
        df_scaled = df.copy() 
        scaler = StandardScaler() 
         
        df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale]) 
         
        print(f"
✓ Scaled {len(features_to_scale)} numeric features") 
        print(f"  Features: {', '.join(features_to_scale[:3])}...") 
        print(f"  Scaler: StandardScaler (mean=0, std=1)") 
         
        return df_scaled, scaler 
     
    def apply_pca(self, df: pd.DataFrame, numeric_features: List[str], 
                 n_components: int = 0.95) -> Tuple[pd.DataFrame, PCA]: 
        """ 
        Apply PCA for dimensionality reduction. 
         
        Why: Reduces multicollinearity and computational complexity. 
         
        Args: 
            df: DataFrame with scaled features 
            numeric_features: List of numeric columns for PCA 
            n_components: Number of components or variance threshold (0.95 = 95%) 
             
        Returns: 
            Tuple of (transformed DataFrame, fitted PCA object) 
        """ 
        print("
" + "=" * 60) 
        print("PRINCIPAL COMPONENT ANALYSIS (PCA)") 
        print("=" * 60) 
         
        pca = PCA(n_components=n_components) 
        pca_transformed = pca.fit_transform(df[numeric_features]) 
         
        # Create DataFrame with PCA components 
        pca_df = pd.DataFrame(pca_transformed, 
                             columns=[f'PC{i+1}' for i in range(pca_transformed.shape[1])]) 
         
        # Add back categorical columns 
        categorical_cols = df.select_dtypes(include=['object']).columns 
        for col in categorical_cols: 
            pca_df[col] = df[col].values 
         
        print(f"
✓ PCA applied successfully") 
        print(f"  Original features: {len(numeric_features)}") 
        print(f"  PCA components: {pca_transformed.shape[1]}") 
        print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.2%}") 
         
        return pca_df, pca 
     
    def get_feature_importance(self, df: pd.DataFrame) -> Dict: 
        """ 
        Generate feature statistics for importance analysis. 
         
        Args: 
            df: DataFrame with engineered features 
             
        Returns: 
            Dictionary with feature statistics 
        """ 
        numeric_cols = df.select_dtypes(include=[np.number]).columns 
         
        feature_stats = {} 
        for col in numeric_cols: 
            feature_stats[col] = { 
                'mean': df[col].mean(), 
                'std': df[col].std(), 
                'min': df[col].min(), 
                'max': df[col].max(), 
                'variance': df[col].var() 
            } 
         
        return feature_stats
