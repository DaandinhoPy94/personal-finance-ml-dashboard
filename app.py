import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Personal Finance ML Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("💰 Personal Finance ML Dashboard")
st.markdown("*Intelligent insights into your spending patterns*")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Upload Data", "Dashboard", "Analytics", "Settings"])

if page == "Upload Data":
    st.header("📊 Data Upload")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your transaction data (CSV)", 
        type=['csv'],
        help="Upload a CSV file with your financial transactions"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} transactions!")
            
            # Show data preview
            st.subheader("Data Preview (eerste 10 rijen)")
            st.dataframe(df.head(10))
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", len(df))
            
            with col2:
                if 'amount' in df.columns:
                    total_spent = df[df['amount'] < 0]['amount'].sum()
                    st.metric("Total Spent", f"€{abs(total_spent):,.2f}")
            
            with col3:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    date_range = df['date'].max() - df['date'].min()
                    st.metric("Date Range", f"{date_range.days} days")
            
            with col4:
                if 'category' in df.columns:
                    unique_categories = df['category'].nunique()
                    st.metric("Categories", unique_categories)
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show expected format
        st.subheader("Expected CSV Format")
        sample_data = {
            'date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'amount': [-12.50, -3.20, -45.00],
            'description': ['Albert Heijn groceries', 'Coffee at work', 'Dinner at restaurant'],
            'category': ['Food', 'Food', 'Food'],
            'subcategory': ['Groceries', 'Coffee', 'Dining_Out'],
            'payment_method': ['Card', 'Card', 'Card']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)

elif page == "Dashboard":
    st.header("📈 Financial Dashboard")
    st.info("🚧 Dashboard will be built in Phase 2 - Data Visualization")

elif page == "Analytics":
    st.header("🤖 ML Analytics")
    st.info("🚧 ML Analytics will be built in Phase 3 - Machine Learning Integration")

elif page == "Settings":
    st.header("⚙️ Settings")
    st.info("🚧 Settings will be added in later phases")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ and Python*")
