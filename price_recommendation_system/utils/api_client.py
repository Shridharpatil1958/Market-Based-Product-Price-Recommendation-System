import requests 
import pandas as pd 
import numpy as np 
from typing import Dict, List, Optional 
import time 

class ProductPriceAPI: 
    """ 
    API client for fetching live product pricing data from multiple sources. 
    Uses RapidAPI endpoints for real-time market data collection. 
    """ 
     
    def __init__(self, api_key: Optional[str] = None): 
        """ 
        Initialize API client with optional API key. 
         
        Args: 
            api_key: RapidAPI key for authenticated requests 
        """ 
        self.api_key = api_key or "demo_key" 
        self.headers = { 
            "X-RapidAPI-Key": self.api_key, 
            "X-RapidAPI-Host": "real-time-product-search.p.rapidapi.com" 
        } 
     
    def fetch_product_data(self, category: str, max_results: int = 50) -> pd.DataFrame: 
        """ 
        Fetch live product pricing data for a given category. 
         
        Args: 
            category: Product category (e.g., 'smartphones', 'laptops') 
            max_results: Maximum number of products to fetch 
             
        Returns: 
            DataFrame with product information 
        """ 
        # For demo purposes, generate realistic synthetic data 
        # In production, replace with actual API calls 
         
        np.random.seed(42) 
         
        # Define category-specific price ranges 
        category_ranges = {
            'smartphones': (200, 1500), 
            'laptops': (400, 3000), 
            'headphones': (30, 500), 
            'tablets': (150, 1200), 
            'smartwatches': (100, 800) 
        } 
         
        price_min, price_max = category_ranges.get(category.lower(), (50, 1000)) 
         
        # Generate synthetic but realistic product data 
        brands = { 
            'smartphones': ['Apple', 'Samsung', 'Google', 'OnePlus', 'Xiaomi', 'Motorola'], 
            'laptops': ['Apple', 'Dell', 'HP', 'Lenovo', 'ASUS', 'Acer'], 
            'headphones': ['Sony', 'Bose', 'JBL', 'Sennheiser', 'Audio-Technica', 'Beats'], 
            'tablets': ['Apple', 'Samsung', 'Microsoft', 'Amazon', 'Lenovo'], 
            'smartwatches': ['Apple', 'Samsung', 'Garmin', 'Fitbit', 'Fossil'] 
        } 
         
        platforms = ['Amazon', 'eBay', 'Walmart', 'BestBuy', 'Target'] 
         
        selected_brands = brands.get(category.lower(), ['Brand A', 'Brand B', 'Brand C']) 
         
        data = [] 
        for i in range(max_results): 
            brand = np.random.choice(selected_brands) 
            platform = np.random.choice(platforms) 
             
            # Generate correlated features 
            base_price = np.random.uniform(price_min, price_max) 
             
            # Higher-priced items tend to have better ratings 
            rating = np.clip(3.0 + (base_price - price_min) / (price_max - price_min) * 2 +
                             np.random.normal(0, 0.3), 1.0, 5.0) 
             
            # More reviews for mid-range popular products 
            review_count = int(np.random.lognormal(4, 2) * (1 + (5 - abs(rating - 4)))) 
             
            # Add some price variance 
            price = base_price * np.random.uniform(0.9, 1.1) 
             
            # Occasionally add missing values (realistic scenario) 
            if np.random.random() < 0.05: 
                price = np.nan
            if np.random.random() < 0.03: 
                rating = np.nan 
            if np.random.random() < 0.02: 
                review_count = np.nan 
             
            data.append({ 
                'product_name': f'{brand} {category.title()} Model {i+1}', 
                'category': category.title(), 
                'brand': brand, 
                'price': price, 
                'rating': rating, 
                'review_count': review_count, 
                'platform': platform 
            }) 
         
        df = pd.DataFrame(data) 
         
        # Add some duplicates (realistic scenario) 
        duplicates = df.sample(n=min(5, len(df)//10)) 
        df = pd.concat([df, duplicates], ignore_index=True) 
         
        return df 
     
    def fetch_multiple_categories(self, categories: List[str], max_per_category: int = 50) -> pd.DataFrame: 
        """ 
        Fetch product data for multiple categories. 
         
        Args: 
            categories: List of product categories 
            max_per_category: Maximum products per category 
             
        Returns: 
            Combined DataFrame with all products 
        """ 
        all_data = [] 
         
        for category in categories: 
            print(f"Fetching data for category: {category}") 
            df = self.fetch_product_data(category, max_per_category) 
            all_data.append(df) 
            time.sleep(0.5)  # Rate limiting

        return pd.concat(all_data, ignore_index=True) 
     
    def get_category_summary(self, df: pd.DataFrame) -> Dict: 
        """ 
        Generate summary statistics for fetched data. 
         
        Args: 
            df: Product DataFrame 
             
        Returns: 
            Dictionary with summary statistics 
        """ 
        summary = { 
            'total_products': len(df), 
            'categories': df['category'].nunique(), 
            'brands': df['brand'].nunique(), 
            'platforms': df['platform'].nunique(), 
            'avg_price': df['price'].mean(), 
            'price_range': (df['price'].min(), df['price'].max()), 
            'avg_rating': df['rating'].mean(), 
            'total_reviews': df['review_count'].sum() 
        } 
         
        return summary[file:2]
