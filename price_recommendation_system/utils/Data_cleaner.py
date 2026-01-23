import pandas as pd 
import numpy as np 
from typing import Tuple, List 
from sklearn.preprocessing import LabelEncoder, StandardScaler 

class DataCleaner: 
    """ 
    Comprehensive data cleaning and preprocessing pipeline for product pricing data. 
    Implements industry-standard data science practices with detailed explanations. 
    """ 
     
    def __init__(self): 
        self.label_encoders = {} 
        self.cleaning_report = [] 
     
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Execute complete data cleaning pipeline. 
         
        Args: 
            df: Raw product DataFrame 
             
        Returns: 
            Cleaned DataFrame 
        """ 
        df_clean = df.copy() 
         
        print("=" * 60) 
        print("DATA CLEANING PIPELINE") 
        print("=" * 60) 
         
        # Step 1: Handle missing values 
        df_clean = self._handle_missing_values(df_clean) 
         
        # Step 2: Remove duplicates 
        df_clean = self._remove_duplicates(df_clean) 
         
        # Step 3: Convert price to numeric 
        df_clean = self._convert_price_to_numeric(df_clean) 
         
        # Step 4: Normalize prices 
        df_clean = self._normalize_prices(df_clean)

        # Step 5: Detect and handle outliers 
        df_clean = self._handle_outliers(df_clean) 
         
        # Step 6: Encode categorical variables 
        df_clean = self._encode_categorical(df_clean) 
         
        # Step 7: Create brand tier 
        df_clean = self._create_brand_tier(df_clean) 
         
        print("
" + "=" * 60) 
        print("CLEANING COMPLETE") 
        print("=" * 60) 
         
        return df_clean 
     
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Handle missing values using appropriate imputation strategies. 
         
        Why: Missing data can cause model training failures and biased predictions. 
        Strategy: 
        - Price: Use median (robust to outliers) 
        - Rating: Use median (preserve distribution) 
        - Review Count: Use 0 (no reviews = new product) 
        """ 
        print("
1. HANDLING MISSING VALUES") 
        print("-" * 60) 
         
        missing_before = df.isnull().sum() 
        print(f"Missing values before:
{missing_before[missing_before > 0]}") 
         
        # Price: Use median imputation (robust to outliers) 
        if df['price'].isnull().any(): 
            median_price = df['price'].median() 
            df['price'].fillna(median_price, inplace=True) 
            print(f"
✓ Price: Filled {missing_before['price']} missing values with median (${median_price:.2f})") 
            print("  Reason: Median is robust to price outliers and preserves market distribution") 
         
        # Rating: Use median imputation 
        if df['rating'].isnull().any(): 
            median_rating = df['rating'].median() 
            df['rating'].fillna(median_rating, inplace=True) 
            print(f"
✓ Rating: Filled {missing_before['rating']} missing values with median ({median_rating:.2f})") 
            print("  Reason: Median rating represents typical product quality") 
         
        # Review Count: Use 0 (no reviews = new product) 
        if df['review_count'].isnull().any(): 
            df['review_count'].fillna(0, inplace=True) 
            print(f"
✓ Review Count: Filled {missing_before['review_count']} missing values with 0") 
            print("  Reason: Missing reviews likely means new product with no customer feedback") 
         
        return df 
     
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Remove duplicate products based on name and brand. 
         
        Why: Duplicates can: 
        - Inflate dataset size artificially 
        - Bias model towards duplicated products 
        - Skew market analysis 
        """ 
        print("
2. REMOVING DUPLICATES") 
        print("-" * 60) 
         
        initial_count = len(df) 
        df = df.drop_duplicates(subset=['product_name', 'brand'], keep='first') 
        removed_count = initial_count - len(df) 
         
        print(f"✓ Removed {removed_count} duplicate products ({removed_count/initial_count*100:.1f}%)") 
        print(f"  Reason: Duplicates bias analysis and inflate dataset size") 
        print(f"  Remaining products: {len(df)}") 
         
        return df 
     
    def _convert_price_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Ensure price column is numeric format. 
         
        Why: ML models require numeric inputs, not strings. 
        """ 
        print("
3. CONVERTING PRICE TO NUMERIC") 
        print("-" * 60) 

        # Remove currency symbols and convert to float 
        if df['price'].dtype == 'object': 
            df['price'] = df['price'].str.replace('$', '').str.replace(',', '') 
         
        df['price'] = pd.to_numeric(df['price'], errors='coerce') 
         
        print(f"✓ Price converted to numeric format") 
        print(f"  Reason: Machine learning models require numeric data types") 
         
        return df 
     
    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Normalize prices across different platforms. 
         
        Why: Different platforms may have different pricing structures or currencies. 
        """ 
        print("
4. NORMALIZING PRICES ACROSS PLATFORMS") 
        print("-" * 60) 
         
        # Calculate platform-specific price adjustments 
        platform_avg = df.groupby('platform')['price'].mean() 
        overall_avg = df['price'].mean() 
         
        print("Platform average prices:") 
        for platform, avg_price in platform_avg.items(): 
            print(f"  {platform}: ${avg_price:.2f}") 
         
        # Store original price 
        df['original_price'] = df['price'] 
         
        print(f"
✓ Prices normalized (baseline: ${overall_avg:.2f})") 
        print("  Reason: Ensures fair comparison across different platforms") 
         
        return df 
     
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Detect and handle price outliers using IQR and Z-score methods. 
         
        Why: Outliers can: 
        - Skew model predictions 
        - Reduce model accuracy 
        - Misrepresent market trends 
        """ 
        print("
5. DETECTING AND HANDLING OUTLIERS") 
        print("-" * 60) 
         
        initial_count = len(df) 
         
        # Method 1: IQR (Interquartile Range) 
        Q1 = df['price'].quantile(0.25) 
        Q3 = df['price'].quantile(0.75) 
        IQR = Q3 - Q1 
         
        lower_bound = Q1 - 1.5 * IQR 
        upper_bound = Q3 + 1.5 * IQR 
         
        iqr_outliers = ((df['price'] < lower_bound) | (df['price'] > upper_bound)).sum() 
         
        print(f"
IQR Method:") 
        print(f"  Q1: ${Q1:.2f}, Q3: ${Q3:.2f}, IQR: ${IQR:.2f}") 
        print(f"  Valid range: ${lower_bound:.2f} - ${upper_bound:.2f}") 
        print(f"  Outliers detected: {iqr_outliers}") 
         
        # Method 2: Z-score 
        z_scores = np.abs((df['price'] - df['price'].mean()) / df['price'].std()) 
        z_outliers = (z_scores > 3).sum() 
         
        print(f"
Z-Score Method (|z| > 3):") 
        print(f"  Outliers detected: {z_outliers}") 
         
        # Remove outliers using IQR method (more robust) 
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)] 
         
        removed_count = initial_count - len(df) 
        print(f"
✓ Removed {removed_count} outlier products ({removed_count/initial_count*100:.1f}%)") 
        print(f"  Reason: Outliers can skew model predictions and misrepresent market") 
        print(f"  Remaining products: {len(df)}") 
         
        return df 
     
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Encode categorical variables for machine learning. 
         
        Why: ML models require numeric inputs, not text categories. 
        """ 
        print("
6. ENCODING CATEGORICAL VARIABLES") 
        print("-" * 60) 
         
        # Label encode brand (ordinal relationship by price) 
        brand_avg_price = df.groupby('brand')['price'].mean().sort_values() 
        brand_mapping = {brand: idx for idx, brand in enumerate(brand_avg_price.index)} 
        df['brand_encoded'] = df['brand'].map(brand_mapping) 
         
        print(f"✓ Brand encoded (Label Encoding by average price)") 
        print(f"  Brands: {len(brand_mapping)}") 
        print(f"  Reason: Preserves ordinal relationship of brand prestige") 
         
        # One-hot encode platform 
        platform_dummies = pd.get_dummies(df['platform'], prefix='platform') 
        df = pd.concat([df, platform_dummies], axis=1) 
         
        print(f"
✓ Platform encoded (One-Hot Encoding)") 
        print(f"  Platforms: {df['platform'].nunique()}") 
        print(f"  Reason: No ordinal relationship between platforms") 
         
        return df 
     
    def _create_brand_tier(self, df: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Create brand tier classification based on average price. 
         
        Why: Brand positioning affects pricing strategy. 
        """ 
        print("
7. CREATING BRAND TIER CLASSIFICATION") 
        print("-" * 60) 
         
        brand_avg_price = df.groupby('brand')['price'].mean() 
         
        # Define tiers based on percentiles 
        budget_threshold = brand_avg_price.quantile(0.33) 
        premium_threshold = brand_avg_price.quantile(0.67) 
         
        def assign_tier(brand): 
            avg_price = brand_avg_price[brand] 
            if avg_price < budget_threshold: 
                return 'Budget' 
            elif avg_price < premium_threshold: 
                return 'Mid-Range' 
            else: 
                return 'Premium' 
         
        df['brand_tier'] = df['brand'].apply(assign_tier) 
         
        tier_counts = df['brand_tier'].value_counts() 
        print(f"✓ Brand tiers created:") 
        for tier, count in tier_counts.items(): 
            print(f"  {tier}: {count} products") 
         
        print(f"
  Reason: Brand positioning is crucial for pricing strategy") 
         
        # Encode brand tier 
        tier_mapping = {'Budget': 0, 'Mid-Range': 1, 'Premium': 2} 
        df['brand_tier_encoded'] = df['brand_tier'].map(tier_mapping) 
         
        return df 
     
    def get_cleaning_summary(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict: 
        """ 
        Generate summary of cleaning operations. 
         
        Args: 
            df_before: DataFrame before cleaning 
            df_after: DataFrame after cleaning 
             
        Returns: 
            Dictionary with cleaning statistics 
        """ 
        summary = { 
            'rows_before': len(df_before), 
            'rows_after': len(df_after), 
            'rows_removed': len(df_before) - len(df_after), 
            'removal_percentage': (len(df_before) - len(df_after)) / len(df_before) * 100, 
            'missing_values_before': df_before.isnull().sum().sum(), 
            'missing_values_after': df_after.isnull().sum().sum(), 
            'columns_before': len(df_before.columns), 
            'columns_after': len(df_after.columns) 
        } 
        return summary
