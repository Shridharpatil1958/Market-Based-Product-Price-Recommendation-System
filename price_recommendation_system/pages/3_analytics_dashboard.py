import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import sys 
import os 

# Add parent directory to path 
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

# Page config 
st.set_page_config(page_title="Analytics Dashboard", page_icon="📈", layout="wide") 

# Set style 
sns.set_style("whitegrid") 
plt.rcParams['figure.figsize'] = (10, 6) 
plt.rcParams['font.size'] = 10 

# Custom CSS 
st.markdown(""" 
<style> 
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 2rem; 
    } 
     
    .chart-card { 
        background: white; 
        padding: 1.5rem; 
        border-radius: 15px; 
        box-shadow: 0 5px 20px rgba(0,0,0,0.1); 
        margin-bottom: 1.5rem; 
    } 
     
    .insight-box { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white; 
        padding: 1.5rem; 
        border-radius: 10px; 
        margin: 1rem 0; 
    } 
</style>
""", unsafe_allow_html=True) 

# Header 
st.markdown(""" 
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  
            padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;"> 
    <h1 style="margin: 0;">📈 Analytics Dashboard</h1> 
    <p style="margin-top: 1rem; opacity: 0.9;">Comprehensive Market Analysis & Visualizations</p> 
</div> 
""", unsafe_allow_html=True) 

# Check if data is available 
data_available = os.path.exists('data/processed/featured_data.csv') 

if not data_available: 
    st.warning("⚠️ No processed data found. Please fetch and process data in the 'Live Market Data' page first.") 
    st.stop() 

# Load data 
df = pd.read_csv('data/processed/featured_data.csv') 

# Overview Metrics 
st.markdown("## 📊 Market Overview") 

col1, col2, col3, col4, col5 = st.columns(5) 

with col1: 
    st.metric("Total Products", len(df)) 

with col2: 
    st.metric("Avg Price", f"${df['price'].mean():.2f}") 

with col3: 
    st.metric("Price Range", f"${df['price'].min():.0f}-${df['price'].max():.0f}") 

with col4: 
    st.metric("Avg Rating", f"{df['rating'].mean():.2f}★") 

with col5: 
    st.metric("Categories", df['category'].nunique())

# Visualization Section 
st.markdown("## 📉 Exploratory Data Analysis") 

# Row 1: Price Distribution and Platform Comparison 
col1, col2 = st.columns(2) 

with col1: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### 💰 Price Distribution") 
     
    fig, ax = plt.subplots(figsize=(8, 5)) 
     
    # Histogram with KDE 
    ax.hist(df['price'], bins=30, alpha=0.7, color='#667eea', edgecolor='black') 
    ax.set_xlabel('Price ($)', fontsize=12) 
    ax.set_ylabel('Frequency', fontsize=12) 
    ax.set_title('Product Price Distribution', fontsize=14, fontweight='bold') 
    ax.grid(alpha=0.3) 
     
    # Add mean and median lines 
    mean_price = df['price'].mean() 
    median_price = df['price'].median() 
    ax.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:.2f}') 
    ax.axvline(median_price, color='green', linestyle='--', linewidth=2, label=f'Median: ${median_price:.2f}') 
    ax.legend() 
     
    st.pyplot(fig) 
    plt.close() 
     
    st.markdown(""" 
    **Insights:** 
    - Distribution shows market price concentration 
    - Mean vs Median indicates skewness 
    - Helps identify competitive price points 
    """) 
     
    st.markdown('</div>', unsafe_allow_html=True) 

with col2: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### 🏪 Average Price by Platform") 
     
    fig, ax = plt.subplots(figsize=(8, 5)) 
     
    platform_avg = df.groupby('platform')['price'].mean().sort_values() 
    colors = plt.cm.viridis(np.linspace(0, 1, len(platform_avg))) 
     
    platform_avg.plot(kind='barh', ax=ax, color=colors) 
    ax.set_xlabel('Average Price ($)', fontsize=12) 
    ax.set_ylabel('Platform', fontsize=12) 
    ax.set_title('Price Comparison Across Platforms', fontsize=14, fontweight='bold') 
    ax.grid(alpha=0.3, axis='x') 
     
    st.pyplot(fig) 
    plt.close() 
     
    st.markdown(""" 
    **Insights:** 
    - Platform pricing variations 
    - Identifies premium vs budget platforms 
    - Guides platform selection strategy 
    """) 
     
    st.markdown('</div>', unsafe_allow_html=True) 

# Row 2: Category Analysis and Price vs Rating 
col1, col2 = st.columns(2) 

with col1: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### 📦 Average Price by Category") 
     
    fig, ax = plt.subplots(figsize=(8, 5)) 
     
    category_avg = df.groupby('category')['price'].mean().sort_values(ascending=False) 
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'][:len(category_avg)] 
     
    category_avg.plot(kind='bar', ax=ax, color=colors) 
    ax.set_xlabel('Category', fontsize=12) 
    ax.set_ylabel('Average Price ($)', fontsize=12) 
    ax.set_title('Price Levels Across Categories', fontsize=14, fontweight='bold') 
    ax.tick_params(axis='x', rotation=45) 
    ax.grid(alpha=0.3, axis='y')
    
    st.pyplot(fig) 
    plt.close() 
     
    st.markdown(""" 
    **Insights:** 
    - Category price hierarchy 
    - Market size and value 
    - Category selection impact on pricing 
    """) 
     
    st.markdown('</div>', unsafe_allow_html=True) 

with col2: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### ⭐ Price vs Rating Relationship") 
     
    fig, ax = plt.subplots(figsize=(8, 5)) 
     
    scatter = ax.scatter(df['rating'], df['price'],  
                        c=df['review_count'], cmap='viridis', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5) 
     
    # Add trend line 
    z = np.polyfit(df['rating'], df['price'], 1) 
    p = np.poly1d(z) 
    ax.plot(df['rating'].sort_values(), p(df['rating'].sort_values()),  
            "r--", linewidth=2, label='Trend') 
     
    ax.set_xlabel('Rating (★)', fontsize=12) 
    ax.set_ylabel('Price ($)', fontsize=12) 
    ax.set_title('Price vs Rating Correlation', fontsize=14, fontweight='bold') 
    ax.grid(alpha=0.3) 
    ax.legend() 
     
    # Add colorbar 
    cbar = plt.colorbar(scatter, ax=ax) 
    cbar.set_label('Review Count', fontsize=10) 
     
    st.pyplot(fig) 
    plt.close() 
     
    correlation = df['price'].corr(df['rating']) 
    st.markdown(f""" 
    **Insights:** 
    - Correlation: {correlation:.3f} 
    - Higher ratings may justify premium pricing 
    - Review count indicates market validation 
    """) 
     
    st.markdown('</div>', unsafe_allow_html=True) 

# Row 3: Review Trends and Brand Analysis 
col1, col2 = st.columns(2) 

with col1: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### 💬 Review Count vs Price Trends") 
     
    fig, ax = plt.subplots(figsize=(8, 5)) 
     
    # Create price bins 
    df['price_bin'] = pd.cut(df['price'], bins=5) 
    review_by_price = df.groupby('price_bin')['review_count'].mean() 
     
    # Convert intervals to strings for plotting 
    x_labels = [f"${int(interval.left)}-${int(interval.right)}" for interval in review_by_price.index] 
     
    ax.plot(range(len(review_by_price)), review_by_price.values,  
            marker='o', linewidth=2, markersize=8, color='#667eea') 
    ax.set_xticks(range(len(review_by_price))) 
    ax.set_xticklabels(x_labels, rotation=45, ha='right') 
    ax.set_xlabel('Price Range ($)', fontsize=12) 
    ax.set_ylabel('Average Review Count', fontsize=12) 
    ax.set_title('Review Popularity by Price Range', fontsize=14, fontweight='bold') 
    ax.grid(alpha=0.3) 
     
    st.pyplot(fig) 
    plt.close() 
     
    st.markdown(""" 
    **Insights:** 
    - Popular price points have more reviews 
    - Sweet spot for market engagement 
    - Indicates demand concentration 
    """)
    
    st.markdown('</div>', unsafe_allow_html=True) 

with col2: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### 🏷️ Top Brands by Average Price") 
     
    fig, ax = plt.subplots(figsize=(8, 5)) 
     
    brand_avg = df.groupby('brand')['price'].mean().sort_values(ascending=False).head(10) 
    colors = plt.cm.plasma(np.linspace(0, 1, len(brand_avg))) 
     
    brand_avg.plot(kind='barh', ax=ax, color=colors) 
    ax.set_xlabel('Average Price ($)', fontsize=12) 
    ax.set_ylabel('Brand', fontsize=12) 
    ax.set_title('Premium vs Budget Brands', fontsize=14, fontweight='bold') 
    ax.grid(alpha=0.3, axis='x') 
     
    st.pyplot(fig) 
    plt.close() 
     
    st.markdown(""" 
    **Insights:** 
    - Brand prestige and pricing power 
    - Premium brand identification 
    - Competitive positioning reference 
    """) 
     
    st.markdown('</div>', unsafe_allow_html=True) 

# Row 4: Competitive Price Bands and Correlation Heatmap 
col1, col2 = st.columns(2) 

with col1: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### 🎯 Competitive Price Bands") 
     
    fig, ax = plt.subplots(figsize=(8, 5)) 
     
    # Box plot by category 
    df.boxplot(column='price', by='category', ax=ax) 
    ax.set_xlabel('Category', fontsize=12) 
    ax.set_ylabel('Price ($)', fontsize=12) 
    ax.set_title('Price Distribution by Category', fontsize=14, fontweight='bold') 
    plt.suptitle('')  # Remove default title 
    ax.tick_params(axis='x', rotation=45) 
    ax.grid(alpha=0.3) 
     
    st.pyplot(fig) 
    plt.close() 
     
    st.markdown(""" 
    **Insights:** 
    - Price quartiles show competitive bands 
    - Outliers indicate premium/budget extremes 
    - Median line shows market center 
    """) 
     
    st.markdown('</div>', unsafe_allow_html=True) 

with col2: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### 🔥 Feature Correlation Heatmap") 
     
    fig, ax = plt.subplots(figsize=(8, 5)) 
     
    # Select numeric columns for correlation 
    numeric_cols = ['price', 'rating', 'review_count', 'demand_score',  
                   'market_avg_price', 'competitor_density', 'brand_position_score'] 
    available_cols = [col for col in numeric_cols if col in df.columns] 
     
    corr_matrix = df[available_cols].corr() 
     
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',  
                center=0, ax=ax, square=True, linewidths=1) 
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold') 
    plt.xticks(rotation=45, ha='right') 
    plt.yticks(rotation=0) 
     
    st.pyplot(fig) 
    plt.close() 
     
    st.markdown(""" 
    **Insights:** 
    - Strong correlations indicate predictive features 
    - Multicollinearity detection 
    - Feature importance indicators 
    """) 
     
    st.markdown('</div>', unsafe_allow_html=True) 

# Brand Tier Analysis 
st.markdown("## 🏆 Brand Tier Analysis") 

col1, col2, col3 = st.columns(3) 

with col1: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### Budget Tier") 
     
    budget_df = df[df['brand_tier'] == 'Budget'] 
    st.metric("Products", len(budget_df)) 
    st.metric("Avg Price", f"${budget_df['price'].mean():.2f}") 
    st.metric("Avg Rating", f"{budget_df['rating'].mean():.2f}★") 
     
    st.markdown('</div>', unsafe_allow_html=True) 

with col2: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### Mid-Range Tier") 
     
    mid_df = df[df['brand_tier'] == 'Mid-Range'] 
    st.metric("Products", len(mid_df)) 
    st.metric("Avg Price", f"${mid_df['price'].mean():.2f}") 
    st.metric("Avg Rating", f"{mid_df['rating'].mean():.2f}★") 
     
    st.markdown('</div>', unsafe_allow_html=True) 

with col3: 
    st.markdown('<div class="chart-card">', unsafe_allow_html=True) 
    st.markdown("### Premium Tier") 
     
    premium_df = df[df['brand_tier'] == 'Premium'] 
    st.metric("Products", len(premium_df)) 
    st.metric("Avg Price", f"${premium_df['price'].mean():.2f}") 
    st.metric("Avg Rating", f"{premium_df['rating'].mean():.2f}★") 
     
    st.markdown('</div>', unsafe_allow_html=True)

# Key Insights Summary 
st.markdown("## 💡 Key Market Insights") 

st.markdown(""" 
<div class="insight-box"> 
    <h3>📊 Market Analysis Summary</h3> 
     
    <h4>Price Distribution:</h4> 
    <ul> 
        <li>Market shows clear price segmentation across tiers</li> 
        <li>Majority of products cluster around median price point</li> 
        <li>Premium segment commands significant price premium</li> 
    </ul> 
     
    <h4>Quality-Price Relationship:</h4> 
    <ul> 
        <li>Positive correlation between rating and price</li> 
        <li>Higher-priced products tend to have better ratings</li> 
        <li>Review count indicates market validation</li> 
    </ul> 
     
    <h4>Competitive Landscape:</h4> 
    <ul> 
        <li>Multiple platforms offer similar products at varying prices</li> 
        <li>Brand positioning significantly impacts pricing power</li> 
        <li>Category selection affects achievable price points</li> 
    </ul> 
     
    <h4>Recommendations:</h4> 
    <ul> 
        <li>Target mid-range segment for balanced risk-reward</li> 
        <li>Invest in quality to justify premium pricing</li> 
        <li>Monitor competitor pricing within your category</li> 
        <li>Build reviews early to establish market credibility</li> 
    </ul> 
</div> 
""", unsafe_allow_html=True) 

# Download Section 
st.markdown("## 📥 Export Data") 

col1, col2 = st.columns(2) 

with col1: 
    csv = df.to_csv(index=False) 
    st.download_button( 
        label="📊 Download Full Dataset", 
        data=csv, 
        file_name="market_analysis_data.csv", 
        mime="text/csv" 
    ) 

with col2: 
    summary_stats = df.describe() 
    summary_csv = summary_stats.to_csv() 
    st.download_button( 
        label="📈 Download Summary Statistics", 
        data=summary_csv, 
        file_name="summary_statistics.csv", 
        mime="text/csv" 
  )
