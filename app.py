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
page = st.sidebar.selectbox("Choose a page", ["Upload Data", "Dashboard", "Analytics", "Portfolio", "Settings"])

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
                        # Show category distribution
                        category_counts = df.dropna(subset=['description', 'category'])['category'].value_counts()
                        
                        st.write("**📊 Categorie Verdeling:**")
                        for category, count in category_counts.items():
                            emoji = "✅" if count >= 2 else "⚠️"
                            st.write(f"{emoji} {category}: {count} transacties")
                        
                        # Check if we have enough categories with sufficient data
                        valid_categories = category_counts[category_counts >= 2]
                        
                        if len(valid_categories) < 2:
                            st.error("❌ Minimaal 2 categorieën met elk minimaal 2 transacties nodig voor ML training")
                            st.info("💡 Tip: Voeg meer sample data toe of combineer kleine categorieën")
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

elif page == "Portfolio":
    st.header("💼 Crypto Portfolio Tracker")
    
    # Import our API modules
    from src.api_integrations import CryptoDataFetcher, PortfolioTracker
    
    # Initialize objects in session state
    if 'crypto_fetcher' not in st.session_state:
        st.session_state.crypto_fetcher = CryptoDataFetcher()
    if 'portfolio_tracker' not in st.session_state:
        st.session_state.portfolio_tracker = PortfolioTracker()
    
    # Create tabs for portfolio features
    portfolio_tab1, portfolio_tab2, portfolio_tab3 = st.tabs(["🔴 Live Prices", "💼 My Portfolio", "📈 Market Analysis"])
    
    with portfolio_tab1:
        st.subheader("🔴 Live Cryptocurrency Prices")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔄 Refresh Prices", type="primary"):
                with st.spinner("📡 Fetching live prices..."):
                    live_prices = st.session_state.crypto_fetcher.get_live_prices()
                    st.session_state.live_crypto_prices = live_prices
            
            # Show live prices if available
            if 'live_crypto_prices' in st.session_state:
                prices = st.session_state.live_crypto_prices
                
                if prices:
                    st.success(f"✅ Live prices updated: {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Create price table
                    price_data = []
                    for coin_id, data in prices.items():
                        price_data.append({
                            'Cryptocurrency': data['name'],
                            'Symbol': data['symbol'],
                            'Price (EUR)': f"€{data['price_eur']:,.2f}",
                            'Price (USD)': f"${data['price_usd']:,.2f}",
                            '24h Change': f"{data['change_24h']:+.1f}%",
                            'Market Cap (EUR)': f"€{data['market_cap_eur']:,.0f}",
                            'Last Updated': data['last_updated'].strftime('%H:%M:%S')
                        })
                    
                    price_df = pd.DataFrame(price_data)
                    st.dataframe(price_df, use_container_width=True)
                    
                    # Highlight biggest movers
                    st.subheader("📊 Biggest Movers (24h)")
                    sorted_by_change = sorted(prices.items(), key=lambda x: abs(x[1]['change_24h']), reverse=True)
                    
                    col_up, col_down = st.columns(2)
                    
                    with col_up:
                        st.write("**🟢 Top Gainers:**")
                        gainers = [p for p in sorted_by_change if p[1]['change_24h'] > 0][:3]
                        for coin_id, data in gainers:
                            st.write(f"• {data['symbol']}: +{data['change_24h']:.1f}%")
                    
                    with col_down:
                        st.write("**🔴 Top Losers:**")
                        losers = [p for p in sorted_by_change if p[1]['change_24h'] < 0][:3]
                        for coin_id, data in losers:
                            st.write(f"• {data['symbol']}: {data['change_24h']:.1f}%")
                
                else:
                    st.error("❌ Failed to fetch live prices. Check your internet connection.")
            else:
                st.info("👆 Click 'Refresh Prices' to get live cryptocurrency data!")
        
        with col2:
            st.write("**📊 Market Overview:**")
            st.write("Real-time cryptocurrency prices powered by CoinGecko API")
            st.write("")
            st.write("**Supported Cryptocurrencies:**")
            supported = st.session_state.crypto_fetcher.supported_coins
            for coin_id, symbol in supported.items():
                st.write(f"• {symbol} - {coin_id.replace('-', ' ').title()}")
    
    with portfolio_tab2:
        st.subheader("💼 My Crypto Portfolio")
    
        # Portfolio management interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**➕ Add Holdings:**")
            
            # Initialize portfolio if not exists
            if 'user_portfolio' not in st.session_state:
                st.session_state.user_portfolio = {}
            
            # Coin selection dropdown
            supported_coins = st.session_state.crypto_fetcher.supported_coins
            coin_options = {f"{symbol} - {coin_id.replace('-', ' ').title()}": coin_id 
                        for coin_id, symbol in supported_coins.items()}
            
            selected_coin_display = st.selectbox(
                "Select Cryptocurrency:",
                options=list(coin_options.keys()),
                help="Choose which cryptocurrency to add to your portfolio"
            )
            
            selected_coin = coin_options[selected_coin_display]
            
            # Amount input
            amount = st.number_input(
                f"Amount of {supported_coins[selected_coin]}:",
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%.4f",
                help="Enter how much of this cryptocurrency you own"
            )
            
            # Add to portfolio button
            col_add, col_remove = st.columns(2)
            
            with col_add:
                if st.button("➕ Add to Portfolio", type="primary"):
                    if amount > 0:
                        st.session_state.user_portfolio[selected_coin] = amount
                        st.success(f"✅ Added {amount} {supported_coins[selected_coin]} to portfolio!")
                    else:
                        st.warning("⚠️ Please enter a valid amount > 0")
            
            with col_remove:
                if st.button("🗑️ Remove from Portfolio"):
                    if selected_coin in st.session_state.user_portfolio:
                        removed_amount = st.session_state.user_portfolio.pop(selected_coin)
                        st.success(f"✅ Removed {removed_amount} {supported_coins[selected_coin]} from portfolio!")
                    else:
                        st.warning("⚠️ This cryptocurrency is not in your portfolio")
        
        with col2:
            st.write("**📊 Current Holdings:**")
            
            if st.session_state.user_portfolio:
                # Display current holdings
                holdings_data = []
                for coin_id, amount in st.session_state.user_portfolio.items():
                    symbol = supported_coins.get(coin_id, coin_id.upper())
                    holdings_data.append({
                        'Cryptocurrency': coin_id.replace('-', ' ').title(),
                        'Symbol': symbol,
                        'Amount': f"{amount:.4f}",
                    })
                
                holdings_df = pd.DataFrame(holdings_data)
                st.dataframe(holdings_df, use_container_width=True, hide_index=True)
                
                # Clear all button
                if st.button("🗑️ Clear All Holdings", type="secondary"):
                    st.session_state.user_portfolio = {}
                    st.success("✅ All holdings cleared!")
                    st.rerun()
            else:
                st.info("📝 No holdings yet. Add some cryptocurrencies to start tracking your portfolio!")
        
        # Portfolio valuation section
        if st.session_state.user_portfolio:
            st.divider()
            st.subheader("💰 Portfolio Valuation")
            
            col1, col2, col3 = st.columns(3)
            
            with col2:
                if st.button("🔄 Calculate Portfolio Value", type="primary"):
                    with st.spinner("💹 Calculating portfolio value..."):
                        # Set holdings in portfolio tracker
                        st.session_state.portfolio_tracker.set_holdings(st.session_state.user_portfolio)
                        
                        # Calculate portfolio value
                        portfolio_stats = st.session_state.portfolio_tracker.calculate_portfolio_value()
                        st.session_state.portfolio_stats = portfolio_stats
            
            # Display portfolio statistics
            if 'portfolio_stats' in st.session_state:
                stats = st.session_state.portfolio_stats
                
                # Key metrics
                st.write("**📈 Portfolio Summary:**")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        "Total Value (EUR)", 
                        f"€{stats['total_value_eur']:,.2f}",
                        help="Current market value of your entire portfolio in Euros"
                    )
                
                with metric_col2:
                    st.metric(
                        "Total Value (USD)", 
                        f"${stats['total_value_usd']:,.2f}",
                        help="Current market value of your entire portfolio in US Dollars"
                    )
                
                with metric_col3:
                    change_24h = stats.get('total_change_24h', 0)
                    delta_color = "normal" if change_24h >= 0 else "inverse"
                    st.metric(
                        "24h Change", 
                        f"{change_24h:+.2f}%",
                        delta=f"{change_24h:+.2f}%",
                        help="Percentage change in your portfolio value over the last 24 hours"
                    )
                
                with metric_col4:
                    change_value = stats['total_value_eur'] * (change_24h / 100)
                    st.metric(
                        "24h Change (EUR)", 
                        f"€{change_value:+,.2f}",
                        delta=f"€{change_value:+,.2f}",
                        help="Absolute change in your portfolio value over the last 24 hours"
                    )
                
                # Detailed holdings breakdown
                if stats.get('holdings_detail'):
                    st.write("**📋 Detailed Holdings:**")
                    
                    detailed_data = []
                    for holding in stats['holdings_detail']:
                        detailed_data.append({
                            'Coin': holding['coin'],
                            'Symbol': holding['symbol'],
                            'Amount': f"{holding['amount']:.4f}",
                            'Price (EUR)': f"€{holding['price_eur']:,.2f}",
                            'Value (EUR)': f"€{holding['value_eur']:,.2f}",
                            '24h Change': f"{holding['change_24h']:+.1f}%",
                            '24h Value Change': f"€{holding['change_value_eur']:+,.2f}"
                        })
                    
                    detailed_df = pd.DataFrame(detailed_data)
                    st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                    
                    # Portfolio composition chart
                    st.write("**🥧 Portfolio Composition:**")
                    
                    # Create pie chart data
                    pie_data = []
                    for holding in stats['holdings_detail']:
                        pie_data.append({
                            'Cryptocurrency': holding['symbol'],
                            'Value': holding['value_eur'],
                            'Percentage': (holding['value_eur'] / stats['total_value_eur']) * 100
                        })
                    
                    pie_df = pd.DataFrame(pie_data)
                    
                    # Create plotly pie chart
                    import plotly.express as px
                    
                    fig_pie = px.pie(
                        pie_df, 
                        values='Value', 
                        names='Cryptocurrency',
                        title="Portfolio Distribution by Value (EUR)",
                        hover_data=['Percentage'],
                        labels={'Value': 'Value (EUR)', 'Percentage': 'Percentage (%)'}
                    )
                    
                    fig_pie.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>' +
                                    'Value: €%{value:,.2f}<br>' +
                                    'Percentage: %{percent}<br>' +
                                    '<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Portfolio performance insights
                    st.write("**💡 Portfolio Insights:**")
                    
                    insights_col1, insights_col2 = st.columns(2)
                    
                    with insights_col1:
                        # Best performing coin
                        best_performer = max(stats['holdings_detail'], key=lambda x: x['change_24h'])
                        st.success(f"🏆 **Best Performer (24h):** {best_performer['symbol']} (+{best_performer['change_24h']:.1f}%)")
                        
                        # Largest holding by value
                        largest_holding = max(stats['holdings_detail'], key=lambda x: x['value_eur'])
                        largest_pct = (largest_holding['value_eur'] / stats['total_value_eur']) * 100
                        st.info(f"📊 **Largest Position:** {largest_holding['symbol']} ({largest_pct:.1f}% of portfolio)")
                    
                    with insights_col2:
                        # Worst performing coin
                        worst_performer = min(stats['holdings_detail'], key=lambda x: x['change_24h'])
                        st.error(f"📉 **Worst Performer (24h):** {worst_performer['symbol']} ({worst_performer['change_24h']:.1f}%)")
                        
                        # Diversification insight
                        num_holdings = len(stats['holdings_detail'])
                        avg_allocation = 100 / num_holdings
                        st.info(f"🎯 **Diversification:** {num_holdings} coins (avg {avg_allocation:.1f}% each)")
                    
                    # Update timestamp
                    st.caption(f"📅 Last updated: {stats['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")

                    # Email alerts section
                    st.divider()
                    st.subheader("📧 Email Alert Configuration")

                    # Import email system
                    from src.email_alerts import PortfolioAlertSystem

                    # Initialize in session state
                    if 'alert_system' not in st.session_state:
                        st.session_state.alert_system = PortfolioAlertSystem()

                    alert_col1, alert_col2 = st.columns(2)

                    with alert_col1:
                        st.write("**Email Setup:**")
                    
                    sender_email = st.text_input(
                        "Gmail Address:", 
                        placeholder="your.email@gmail.com",
                        help="Your Gmail address for sending alerts"
                    )
                    
                    sender_password = st.text_input(
                        "Gmail App Password:", 
                        type="password",
                        placeholder="Enter your Gmail app password",
                        help="Generate an app password in Gmail settings → Security → 2FA → App passwords"
                    )
                    
                    recipient_email = st.text_input(
                        "Alert Recipient:", 
                        placeholder="alerts@yourdomain.com",
                        value=sender_email,
                        help="Where to send portfolio alerts (can be same as sender)"
                    )

                    with alert_col2:
                        st.write("**Alert Settings:**")
                    
                    if sender_email and sender_password and recipient_email:
                        # Configure email system
                        st.session_state.alert_system.setup_email_config(
                            sender_email, sender_password, recipient_email
                        )
                        
                        # Test email button
                        if st.button("📧 Send Test Email", type="secondary"):
                            with st.spinner("Sending test email..."):
                                success = st.session_state.alert_system.test_email_setup()
                                if success:
                                    st.success("✅ Test email sent! Check your inbox.")
                                else:
                                    st.error("❌ Email test failed. Check your settings.")
                        
                        # Simulate portfolio alert button  
                        if st.button("🚨 Simulate Portfolio Alert", type="secondary"):
                            # Create fake alert data for demonstration
                            fake_alert = {
                                'should_alert': True,
                                'change_pct': 15.7,
                                'change_eur': 2450.00,
                                'current_value': 18000.00,
                                'previous_value': 15550.00,
                                'alert_type': 'major',
                                'urgency': 'MEDIUM'
                            }
                            
                            with st.spinner("Sending portfolio alert..."):
                                success = st.session_state.alert_system.send_portfolio_alert(
                                    fake_alert, stats
                                )
                                if success:
                                    st.success("✅ Portfolio alert sent! Check your email for the detailed report.")
                                else:
                                    st.error("❌ Alert sending failed. Check your email settings.")
                    else:
                        st.info("💡 Fill in email settings to enable alerts")
                        st.write("**How to get Gmail App Password:**")
                        st.write("1. Go to Google Account Settings")
                        st.write("2. Security → 2-Step Verification (enable if not done)")
                        st.write("3. App passwords → Generate password")
                        st.write("4. Use the generated password (not your regular password)")
                        
                    with portfolio_tab3:
                        st.subheader("📈 Market Analysis")
                        st.info("🚧 Technical analysis charts coming soon!")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ and Python*")
