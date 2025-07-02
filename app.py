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

            # Store in session state for dashboard
            st.session_state.df = df

            st.success(f"✅ Successfully loaded {len(df)} transactions!")
            
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
    
    # Check if we have data
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Geen data gevonden. Upload eerst je transacties in de 'Upload Data' tab.")
        st.info("👆 Gebruik de navigatie in de sidebar, of klik op de button hieronder.")
        if st.button("🔄 Ga naar Upload Data"):
            st.switch_page("pages/upload.py")
    else:
        df = st.session_state.df.copy()
        
        # Data preprocessing
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        df['day_of_week'] = df['date'].dt.day_name()
        df['is_weekend'] = df['date'].dt.weekday >= 5
        
        # Sidebar filters
        st.sidebar.header("📊 Dashboard Filters")
        
        # Date range filter
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Selecteer datumbereik",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Category filter
        categories = ['Alle categorieën'] + list(df['category'].unique())
        selected_category = st.sidebar.selectbox("Selecteer categorie", categories)
        
        # Apply filters
        if len(date_range) == 2:
            df_filtered = df[(df['date'].dt.date >= date_range[0]) & 
                           (df['date'].dt.date <= date_range[1])]
        else:
            df_filtered = df
            
        if selected_category != 'Alle categorieën':
            df_filtered = df_filtered[df_filtered['category'] == selected_category]
        
        # Key Performance Indicators
        st.subheader("💰 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_spent = df_filtered[df_filtered['amount'] < 0]['amount'].sum()
            st.metric("Totale Uitgaven", f"€{abs(total_spent):,.2f}")
        
        with col2:
            total_income = df_filtered[df_filtered['amount'] > 0]['amount'].sum()
            st.metric("Totale Inkomsten", f"€{total_income:,.2f}")
        
        with col3:
            net_amount = total_income + total_spent
            st.metric("Netto Resultaat", f"€{net_amount:,.2f}", 
                     delta=f"€{net_amount:,.2f}")
        
        with col4:
            avg_transaction = df_filtered[df_filtered['amount'] < 0]['amount'].mean()
            st.metric("Gem. Uitgave", f"€{abs(avg_transaction):,.2f}")
        
        # Charts Row 1: Time Series and Category Distribution
        st.subheader("📈 Uitgaven Analyse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily spending chart
            daily_spending = df_filtered[df_filtered['amount'] < 0].groupby('date')['amount'].sum().abs()
            
            fig_time = px.line(
                x=daily_spending.index, 
                y=daily_spending.values,
                title="Dagelijkse Uitgaven",
                labels={'x': 'Datum', 'y': 'Uitgaven (€)'}
            )
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Category pie chart
            category_spending = df_filtered[df_filtered['amount'] < 0].groupby('category')['amount'].sum().abs()
            
            fig_pie = px.pie(
                values=category_spending.values,
                names=category_spending.index,
                title="Uitgaven per Categorie"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Charts Row 2: Spending Patterns
        st.subheader("🔍 Uitgavenpatronen")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending by day of week
            dow_spending = df_filtered[df_filtered['amount'] < 0].groupby('day_of_week')['amount'].sum().abs()
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_spending = dow_spending.reindex(day_order, fill_value=0)
            
            fig_dow = px.bar(
                x=dow_spending.index,
                y=dow_spending.values,
                title="Uitgaven per Weekdag",
                labels={'x': 'Weekdag', 'y': 'Uitgaven (€)'}
            )
            fig_dow.update_layout(height=400)
            st.plotly_chart(fig_dow, use_container_width=True)
        
        with col2:
            # Weekend vs Weekday comparison
            weekend_spending = df_filtered[df_filtered['amount'] < 0].groupby('is_weekend')['amount'].sum().abs()
            weekend_labels = ['Doordeweeks', 'Weekend']
            
            fig_weekend = px.bar(
                x=weekend_labels,
                y=[weekend_spending.get(False, 0), weekend_spending.get(True, 0)],
                title="Weekend vs Doordeweeks Uitgaven",
                labels={'x': 'Periode', 'y': 'Uitgaven (€)'}
            )
            fig_weekend.update_layout(height=400)
            st.plotly_chart(fig_weekend, use_container_width=True)
        
        # Detailed Category Analysis
        st.subheader("📊 Gedetailleerde Categorie Analyse")
        
        # Top subcategories
        subcategory_spending = df_filtered[df_filtered['amount'] < 0].groupby('subcategory')['amount'].sum().abs().sort_values(ascending=False).head(10)
        
        fig_subcat = px.bar(
            x=subcategory_spending.values,
            y=subcategory_spending.index,
            orientation='h',
            title="Top 10 Subcategorieën",
            labels={'x': 'Uitgaven (€)', 'y': 'Subcategorie'}
        )
        fig_subcat.update_layout(height=500)
        st.plotly_chart(fig_subcat, use_container_width=True)
        
        # Recent transactions table
        st.subheader("📋 Recente Transacties")
        recent_transactions = df_filtered.sort_values('date', ascending=False).head(10)
        
        # Format the display
        display_df = recent_transactions[['date', 'amount', 'description', 'category', 'subcategory']].copy()
        display_df['amount'] = display_df['amount'].apply(lambda x: f"€{x:,.2f}")
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True)

elif page == "Analytics":
    st.header("🤖 ML Analytics")
    
    # Check if we have data
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Geen data gevonden. Upload eerst je transacties in de 'Upload Data' tab.")
        st.info("👆 Gebruik de navigatie in de sidebar om naar 'Upload Data' te gaan.")
    else:
        df = st.session_state.df.copy()
        
        # Import our ML models
        from src.ml_models import TransactionCategorizer, AnomalyDetector
        
        # Initialize ML objects
        if 'categorizer' not in st.session_state:
            st.session_state.categorizer = TransactionCategorizer()
        if 'anomaly_detector' not in st.session_state:
            st.session_state.anomaly_detector = AnomalyDetector()
        
        # Create tabs for different ML features
        ml_tab1, ml_tab2, ml_tab3 = st.tabs(["🎯 Auto Categorization", "🚨 Anomaly Detection", "📊 ML Insights"])
        
        with ml_tab1:
            st.subheader("🤖 Automatische Categorisatie")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Hoe het werkt:**")
                st.write("- AI leert van je bestaande transacties en categorieën")
                st.write("- Model voorspelt automatisch categorieën voor nieuwe transacties")
                st.write("- Confidence score toont hoe zeker het model is")
                
                # Check data requirements
                required_cols = ['description', 'category']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missende kolommen voor ML: {missing_cols}")
                else:
                    # Data statistics
                    total_transactions = len(df)
                    categorized_transactions = len(df.dropna(subset=['description', 'category']))
                    unique_categories = df['category'].nunique()
                    
                    st.info(f"📊 **Data Status:**")
                    st.write(f"- Totaal transacties: {total_transactions}")
                    st.write(f"- Bruikbaar voor training: {categorized_transactions}")
                    st.write(f"- Unieke categorieën: {unique_categories}")
                    
                    if categorized_transactions < 10:
                        st.warning("⚠️ Minimaal 10 complete transacties nodig voor training")
                    else:
                        # Training section
                        if st.button("🎓 Train ML Model", type="primary"):
                            with st.spinner("🤖 AI aan het trainen..."):
                                try:
                                    # Train the model
                                    results = st.session_state.categorizer.train(df)
                                    
                                    # Store results in session state
                                    st.session_state.training_results = results
                                    
                                    st.success("✅ Model succesvol getraind!")
                                    
                                    # Show training results
                                    col_acc, col_samples, col_cats = st.columns(3)
                                    with col_acc:
                                        st.metric("Accuracy", f"{results['accuracy']:.1%}")
                                    with col_samples:
                                        st.metric("Training Samples", results['training_samples'])
                                    with col_cats:
                                        st.metric("Categories", len(results['categories']))
                                    
                                except Exception as e:
                                    st.error(f"❌ Training error: {str(e)}")
            
            with col2:
                # Model status
                if st.session_state.categorizer.is_trained:
                    st.success("✅ Model Status: Trained")
                    
                    # Show categories the model knows
                    st.write("**Geleerde Categorieën:**")
                    for cat in st.session_state.categorizer.categories:
                        st.write(f"• {cat}")
                    
                    # Test prediction interface
                    st.write("---")
                    st.write("**🔮 Test Voorspelling:**")
                    
                    test_description = st.text_input(
                        "Typ een beschrijving:",
                        placeholder="bijv. 'Albert Heijn boodschappen'"
                    )
                    
                    if test_description:
                        try:
                            category, confidence = st.session_state.categorizer.predict_category(test_description)
                            
                            st.write(f"**Voorspelling:** `{category}`")
                            st.write(f"**Confidence:** {confidence:.1%}")
                            
                            # Confidence color coding
                            if confidence > 0.8:
                                st.success("🎯 Hoge zekerheid")
                            elif confidence > 0.6:
                                st.warning("🤔 Gemiddelde zekerheid")
                            else:
                                st.error("❓ Lage zekerheid")
                                
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                else:
                    st.info("⏳ Model Status: Not Trained")
                    st.write("Train eerst het model om voorspellingen te kunnen maken.")
        
        with ml_tab2:
            st.subheader("🚨 Anomaly Detection")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Hoe het werkt:**")
                st.write("- Analyseert je normale uitgavenpatronen per categorie")
                st.write("- Detecteert transacties die significant afwijken")
                st.write("- Gebruikt statistische Z-score methode")
                
                # Anomaly detection controls
                threshold = st.slider(
                    "Anomalie gevoeligheid (Z-score threshold):",
                    min_value=1.5, max_value=4.0, value=2.5, step=0.1,
                    help="Lagere waarde = meer anomalieën detecteren"
                )
                
                if st.button("🔍 Detecteer Anomalieën", type="primary"):
                    with st.spinner("🕵️ Zoeken naar ongewone uitgaven..."):
                        try:
                            # Analyze spending patterns
                            spending_stats = st.session_state.anomaly_detector.analyze_spending_patterns(df)
                            
                            # Detect anomalies
                            anomalies = st.session_state.anomaly_detector.detect_anomalies(df, threshold)
                            
                            st.session_state.anomalies = anomalies
                            st.session_state.spending_stats = spending_stats
                            
                            if len(anomalies) > 0:
                                st.warning(f"🚨 {len(anomalies)} anomalieën gevonden!")
                                
                                # Show anomalies table
                                display_anomalies = anomalies[['date', 'amount', 'description', 'category', 'z_score']].copy()
                                display_anomalies['amount'] = display_anomalies['amount'].apply(lambda x: f"€{x:.2f}")
                                display_anomalies['z_score'] = display_anomalies['z_score'].apply(lambda x: f"{x:.1f}")
                                display_anomalies['date'] = pd.to_datetime(display_anomalies['date']).dt.strftime('%Y-%m-%d')
                                
                                st.dataframe(display_anomalies, use_container_width=True)
                            else:
                                st.success("✅ Geen anomalieën gevonden - je uitgaven zijn consistent!")
                                
                        except Exception as e:
                            st.error(f"❌ Anomaly detection error: {str(e)}")
            
            with col2:
                # Show spending statistics if available
                if 'spending_stats' in st.session_state:
                    st.write("**📊 Uitgaven Statistieken:**")
                    
                    for category, stats in st.session_state.spending_stats.items():
                        with st.expander(f"{category}"):
                            st.write(f"Gemiddeld: €{stats['mean']:.2f}")
                            st.write(f"Standaard dev: €{stats['std']:.2f}")
                            st.write(f"Maximum: €{stats['max']:.2f}")
                            st.write(f"Aantal: {stats['count']}")
        
        with ml_tab3:
            st.subheader("📊 ML Model Insights")
            
            # Show training results if available
            if 'training_results' in st.session_state:
                results = st.session_state.training_results
                
                st.write("**🎯 Model Performance:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Overall metrics
                    st.metric("Overall Accuracy", f"{results['accuracy']:.1%}")
                    st.metric("Training Samples", results['training_samples'])
                    st.metric("Test Samples", results['test_samples'])
                
                with col2:
                    # Per-category performance
                    if 'classification_report' in results:
                        st.write("**Per-Category Performance:**")
                        
                        report = results['classification_report']
                        
                        # Create performance dataframe
                        perf_data = []
                        for category in results['categories']:
                            if category in report:
                                perf_data.append({
                                    'Category': category,
                                    'Precision': f"{report[category]['precision']:.1%}",
                                    'Recall': f"{report[category]['recall']:.1%}",
                                    'F1-Score': f"{report[category]['f1-score']:.1%}",
                                    'Support': report[category]['support']
                                })
                        
                        if perf_data:
                            perf_df = pd.DataFrame(perf_data)
                            st.dataframe(perf_df, use_container_width=True)
            else:
                st.info("📈 Train eerst een model om insights te zien.")

elif page == "Settings":
    st.header("⚙️ Settings")
    st.info("🚧 Settings will be added in later phases")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ and Python*")
