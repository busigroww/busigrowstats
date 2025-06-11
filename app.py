import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import io
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Sales Forecasting ML App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.image("logo.png", width=150)

# Title and description
st.markdown('<h1 class="main-header">📊 Sales Forecasting ML Application</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Upload your pre-processed CSV file to automatically apply ML models and generate sales statistics and visualizations.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a pre-processed CSV file with sales data"
)
st.markdown('</div>', unsafe_allow_html=True)

# Helper functions
def create_features(product_data):
    """Create features for a single product"""
    product_data = product_data.sort_values('Date')
    
    # Add lag features (previous 1, 2, 3, 7 days)
    product_data['Sales_Lag1'] = product_data['Total Sales'].shift(1)
    product_data['Sales_Lag2'] = product_data['Total Sales'].shift(2)
    product_data['Sales_Lag3'] = product_data['Total Sales'].shift(3)
    product_data['Sales_Lag7'] = product_data['Total Sales'].shift(7)
    
    # Add rolling mean features (last 3, 7 days)
    product_data['Sales_Rolling3'] = product_data['Total Sales'].rolling(window=3).mean()
    product_data['Sales_Rolling7'] = product_data['Total Sales'].rolling(window=7).mean()
    
    # Add day of week as cyclical features
    product_data['Day_sin'] = np.sin(2 * np.pi * product_data['Day']/7)
    product_data['Day_cos'] = np.cos(2 * np.pi * product_data['Day']/7)
    
    # Add month as cyclical features (for seasonality)
    product_data['Month'] = product_data['Date'].dt.month
    product_data['Month_sin'] = np.sin(2 * np.pi * product_data['Month']/12)
    product_data['Month_cos'] = np.cos(2 * np.pi * product_data['Month']/12)
    
    # Drop rows with NaN values (from lag features)
    product_data = product_data.dropna()
    
    return product_data

def train_product_model(product_name, product_data, min_samples=5):
    """Train model for a single product"""
    
    # Check if we have enough data points
    if len(product_data) < min_samples:
        return None, None, None
    
    # Create features
    product_data = create_features(product_data)
    
    if len(product_data) < min_samples:
        return None, None, None
    
    # Define features and target
    features = ['Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 'Sales_Lag7', 
                'Sales_Rolling3', 'Sales_Rolling7', 'Day_sin', 'Day_cos', 
                'Month_sin', 'Month_cos', 'Temperature', 'Holiday(0/1)']
    
    X = product_data[features]
    y = product_data['Total Sales']
    
    # Use time series split for validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # If we have very few samples, use simple train/test split
    if len(product_data) < 20:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return model, feature_importance, {'mse': mse, 'mae': mae, 'r2': r2}
    
    # For products with more data, use time series cross-validation
    else:
        best_model = None
        best_score = float('inf')
        best_metrics = None
        
        # Try different models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        for model_name, model in models.items():
            cv_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                cv_scores.append(mse)
            
            avg_mse = np.mean(cv_scores)
            
            if avg_mse < best_score:
                best_score = avg_mse
                best_model = model
                
                # Retrain on full dataset
                best_model.fit(X, y)
                
                # Calculate metrics on full dataset predictions (for reference)
                y_pred = best_model.predict(X)
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                best_metrics = {'mse': mse, 'mae': mae, 'r2': r2}
        
        # Get feature importance for the best model
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            # For linear regression, use coefficients
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': np.abs(best_model.coef_)
            }).sort_values('Importance', ascending=False)
        
        return best_model, feature_importance, best_metrics

def create_visualizations(df):
    """Create visualizations similar to the provided examples"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("tab10")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sales Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Top 10 products by sales
    top_products = df.groupby('Product Name')['Total Sales'].sum().sort_values(ascending=False).head(10)
    axes[0, 0].barh(range(len(top_products)), top_products.values, color='steelblue')
    axes[0, 0].set_yticks(range(len(top_products)))
    axes[0, 0].set_yticklabels(top_products.index)
    axes[0, 0].set_xlabel('Total Sales')
    axes[0, 0].set_title('Top 10 Products by Total Sales')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. Temperature vs Sales
    axes[0, 1].scatter(df['Temperature'], df['Total Sales'], alpha=0.6, color='steelblue')
    axes[0, 1].set_xlabel('Temperature')
    axes[0, 1].set_ylabel('Total Sales')
    axes[0, 1].set_title('Temperature vs Total Sales')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Sales by holiday
    holiday_sales = df.groupby('Holiday(0/1)')['Total Sales'].sum()
    axes[1, 0].bar(holiday_sales.index, holiday_sales.values, color='steelblue')
    axes[1, 0].set_xlabel('Holiday (0=No, 1=Yes)')
    axes[1, 0].set_ylabel('Total Sales')
    axes[1, 0].set_title('Total Sales: Holidays vs Non-Holidays')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Sales by day of week
    day_sales = df.groupby('Day')['Total Sales'].sum()
    axes[1, 1].bar(day_sales.index, day_sales.values, color='steelblue')
    axes[1, 1].set_xlabel('Day (1=Monday, 7=Sunday)')
    axes[1, 1].set_ylabel('Total Sales')
    axes[1, 1].set_title('Total Sales by Day of Week')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main application logic
if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Display basic information
        st.success("✅ File uploaded successfully!")
        
        # Data overview section
        st.markdown('<h2 class="sub-header">📋 Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Total Records</h3>
                <h2>{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🛍️ Unique Products</h3>
                <h2>{df['Product Name'].nunique():,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_sales = df['Total Sales'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Total Sales</h3>
                <h2>{total_sales:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_sales = df['Total Sales'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Average Sales</h3>
                <h2>{avg_sales:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Convert Date to datetime if it's not already
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Display date range
            if df['Date'].notna().any():
                date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
                st.info(f"📅 Date Range: {date_range}")
        
        # Display sample data
        st.markdown('<h2 class="sub-header">🔍 Sample Data</h2>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data preprocessing and model training section
        st.markdown('<h2 class="sub-header">🤖 ML Model Training</h2>', unsafe_allow_html=True)
        
        # Configuration options
        col1, col2 = st.columns(2)
        with col1:
            min_samples = st.slider("Minimum samples required per product", 3, 20, 5)
        with col2:
            max_products = st.slider("Maximum products to model", 5, 50, 20)
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training ML models..."):
                # Get list of unique products
                products = df['Product Name'].unique()[:max_products]
                
                # Train models for products
                results = {}
                all_metrics = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, product in enumerate(products):
                    status_text.text(f"Training model for: {product}")
                    
                    # Filter data for this product
                    product_data = df[df['Product Name'] == product].copy()
                    
                    # Train model
                    model, importance, metrics = train_product_model(product, product_data, min_samples)
                    
                    if model is not None:
                        results[product] = {
                            'model': model,
                            'importance': importance,
                            'metrics': metrics
                        }
                        
                        all_metrics.append({
                            'Product': product,
                            'MSE': metrics['mse'],
                            'MAE': metrics['mae'],
                            'R2': metrics['r2']
                        })
                    
                    progress_bar.progress((i + 1) / len(products))
                
                status_text.text("Model training completed!")
                
                # Display results
                if all_metrics:
                    st.success(f"✅ Successfully trained models for {len(results)} out of {len(products)} products")
                    
                    # Model performance metrics
                    st.markdown('<h3 class="sub-header">📊 Model Performance</h3>', unsafe_allow_html=True)
                    
                    metrics_df = pd.DataFrame(all_metrics)
                    
                    # Average metrics
                    avg_metrics = metrics_df[['MSE', 'MAE', 'R2']].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average MSE", f"{avg_metrics['MSE']:.2f}")
                    with col2:
                        st.metric("Average MAE", f"{avg_metrics['MAE']:.2f}")
                    with col3:
                        st.metric("Average R²", f"{avg_metrics['R2']:.3f}")
                    
                    # Detailed metrics table
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Feature importance for best performing model
                    best_product = metrics_df.loc[metrics_df['R2'].idxmax(), 'Product']
                    st.markdown(f'<h3 class="sub-header">🎯 Feature Importance (Best Model: {best_product})</h3>', unsafe_allow_html=True)
                    
                    importance_df = results[best_product]['importance']
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax)
                    ax.set_title('Top 10 Feature Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                else:
                    st.warning("⚠️ No products had sufficient data for modeling. Try reducing the minimum samples requirement.")
        
        # Visualizations section
        st.markdown('<h2 class="sub-header">📈 Sales Analysis Visualizations</h2>', unsafe_allow_html=True)
        
        if st.button("📊 Generate Visualizations", type="secondary"):
            with st.spinner("Creating visualizations..."):
                try:
                    fig = create_visualizations(df)
                    st.pyplot(fig)
                    
                    # Additional individual charts
                    st.markdown('<h3 class="sub-header">📋 Individual Charts</h3>', unsafe_allow_html=True)
                    
                    # Top products chart
                    st.markdown("**Top 10 Products by Sales**")
                    top_products = df.groupby('Product Name')['Total Sales'].sum().sort_values(ascending=False).head(10)
                    st.bar_chart(top_products)
                    
                    # Sales by day
                    st.markdown("**Sales by Day of Week**")
                    day_sales = df.groupby('Day')['Total Sales'].sum()
                    st.bar_chart(day_sales)
                    
                    # Temperature vs Sales scatter plot
                    st.markdown("**Temperature vs Sales Relationship**")
                    chart_data = df[['Temperature', 'Total Sales']].dropna()
                    st.scatter_chart(chart_data.set_index('Temperature'))
                    
                except Exception as e:
                    st.error(f"Error creating visualizations: {str(e)}")
        
        # Download section
        st.markdown('<h2 class="sub-header">💾 Download Results</h2>', unsafe_allow_html=True)
        
        # Prepare summary statistics
        summary_stats = {
            'Total Records': len(df),
            'Unique Products': df['Product Name'].nunique(),
            'Total Sales': df['Total Sales'].sum(),
            'Average Sales': df['Total Sales'].mean(),
            'Date Range': f"{df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else 'N/A'
        }
        
        summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
        
        # Convert to CSV for download
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Summary Statistics",
            data=csv_data,
            file_name="sales_summary_statistics.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the required columns: Product Name, Total Sales, Date, Temperature, Holiday(0/1), Day")

else:
    # Instructions when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background-color: #301934; border-radius: 10px; margin: 2rem 0;">
        <h3>🚀 Get Started</h3>
        <p>Upload a pre-processed CSV file to begin the analysis. Your file should contain the following columns:</p>
        <ul style="text-align: left; display: inline-block;">
            <li><strong>Product Name</strong> - Name of the product has to be mentioned</li>
            <li><strong>Total Sales</strong> - Sales amount</li>
            <li><strong>Date</strong> - Transaction date</li>
            <li><strong>Temperature</strong> - Temperature on that day</li>
            <li><strong>Holiday(0/1)</strong> - Holiday indicator (0=No, 1=Yes)</li>
            <li><strong>Day</strong> - Day of week (1=Monday, 7=Sunday)</li>
        </ul>
        <p>The app will automatically:</p>
        <ul style="text-align: left; display: inline-block;">
            <li>✅ Train ML models for sales forecasting</li>
            <li>📊 Generate comprehensive visualizations</li>
            <li>📈 Provide performance metrics</li>
            <li>💾 Allow you to download results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>📊 Sales Forecasting ML Application | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

