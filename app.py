import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="NovaTrade AI - Predictive Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #E0E0E0;
    }
    .stMetric {
        background-color: #1E2329;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2D3748;
    }
    .stMetric label {
        color: #A0AEC0 !important;
    }
    .stMetric .css-1wivap2 { 
        color: #4FD1C5 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1A202C;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA PREPARATION ---
@st.cache_data
def load_and_preprocess_data():
    try:
        # 1. Load Data
        fg_df = pd.read_csv('fear_greed_index.csv')
        hist_df = pd.read_csv('historical_data.csv')
        
        # 2. Process Dates
        fg_df['date'] = pd.to_datetime(fg_df['date'])
        
        hist_df['Timestamp IST'] = pd.to_datetime(hist_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
        hist_df['date'] = hist_df['Timestamp IST'].dt.floor('D')
        
        # 3. Clean numeric columns
        hist_df['Closed PnL'] = pd.to_numeric(hist_df['Closed PnL'], errors='coerce').fillna(0)
        hist_df['Size USD'] = pd.to_numeric(hist_df['Size USD'], errors='coerce').fillna(0)
        
        # 4. Merge Data (Daily Level)
        merged_df = pd.merge(hist_df, fg_df, on='date', how='inner')
        return merged_df, fg_df, hist_df
    except FileNotFoundError:
        st.error("Error: Could not find 'fear_greed_index.csv' or 'historical_data.csv'. Please ensure they are in the same directory.")
        st.stop()
        return None, None, None

def engineer_features(merged_df):
    # Daily aggregation per trader
    daily_trader = merged_df.groupby(['date', 'Account']).agg({
        'Closed PnL': 'sum',
        'Size USD': 'mean',
        'classification': 'first', # Sentiment for that day
        'value': 'first' # Fear greed numeric value
    }).reset_index()
    
    # Trade counts
    trade_counts = merged_df.groupby(['date', 'Account']).size().reset_index(name='num_trades')
    daily_trader = pd.merge(daily_trader, trade_counts, on=['date', 'Account'])
    
    # Sort for lagging
    daily_trader = daily_trader.sort_values(by=['Account', 'date'])
    
    # Create Lag Features
    daily_trader['prev_day_pnl'] = daily_trader.groupby('Account')['Closed PnL'].shift(1)
    daily_trader['prev_day_sentiment'] = daily_trader.groupby('Account')['value'].shift(1)
    
    # Create Target (Next day profitability) -> 1 if profitable > $0, else 0
    daily_trader['next_day_pnl'] = daily_trader.groupby('Account')['Closed PnL'].shift(-1)
    daily_trader['target_profitable'] = (daily_trader['next_day_pnl'] > 0).astype(int)
    
    # Drop NaNs created by lagging (first/last days per account)
    ml_df = daily_trader.dropna(subset=['prev_day_pnl', 'prev_day_sentiment', 'next_day_pnl'])
    return daily_trader, ml_df

def cluster_traders(merged_df):
    # Aggregate over all time for clustering behavioral archetypes
    trader_profile = merged_df.groupby('Account').agg({
        'Closed PnL': 'sum',
        'Size USD': 'mean'
    }).reset_index()
    
    trade_counts = merged_df.groupby('Account').size().reset_index(name='total_trades')
    trader_profile = pd.merge(trader_profile, trade_counts, on='Account')
    
    # Win rate
    trades_with_pnl = merged_df[merged_df['Closed PnL'] != 0]
    win_rate_df = trades_with_pnl.assign(is_win=trades_with_pnl['Closed PnL'] > 0).groupby('Account')['is_win'].mean().reset_index()
    win_rate_df.rename(columns={'is_win': 'win_rate'}, inplace=True)
    trader_profile = pd.merge(trader_profile, win_rate_df, on='Account', how='left').fillna(0)
    
    # Clustering using K-Means
    features = ['Size USD', 'total_trades', 'win_rate']
    X = trader_profile[features]
    
    # Normalizing just for the K-Means algo internally
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    trader_profile['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Label clusters based on characteristics
    cluster_stats = trader_profile.groupby('cluster')[['Size USD', 'total_trades', 'win_rate']].mean()
    labels = {}
    for i in range(3):
        if cluster_stats.loc[i, 'Size USD'] == cluster_stats['Size USD'].max():
            labels[i] = "High-Volume Archetype"
        elif cluster_stats.loc[i, 'total_trades'] == cluster_stats['total_trades'].max():
            labels[i] = "High-Frequency Archetype"
        else:
             labels[i] = "Cautious/Low-Volume Archetype"
             
    # Fallback to prevent duplicate labels just in case
    if len(set(labels.values())) < 3:
        labels = {0: "Archetype 0", 1: "Archetype 1", 2: "Archetype 2"}
             
    trader_profile['Archetype Name'] = trader_profile['cluster'].map(labels)
    
    return trader_profile

# --- MACHINE LEARNING MODEL ---
@st.cache_resource
def train_predictive_model(ml_df):
    features = ['value', 'Closed PnL', 'num_trades', 'Size USD', 'prev_day_pnl', 'prev_day_sentiment']
    X = ml_df[features]
    y = ml_df['target_profitable']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    # Feature Importance
    importance = clf.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=True)
    
    return clf, acc, report, feat_imp_df

# --- LOADING THE DATA & MODELS ---
with st.spinner("Loading Data and Training Models..."):
    merged_df, fg_df, hist_df = load_and_preprocess_data()
    if merged_df is not None:
        daily_trader, ml_df = engineer_features(merged_df)
        trader_profile = cluster_traders(merged_df)
        clf, accuracy, class_report, feature_importance = train_predictive_model(ml_df)
        
        # Merge archetype back to daily data for filtering
        merged_df = pd.merge(merged_df, trader_profile[['Account', 'Archetype Name']], on='Account', how='left')


# --- SIDEBAR UI ---
if merged_df is not None:
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg", width=60)
    st.sidebar.title("Filters")
    
    date_min = merged_df['date'].min().date()
    date_max = merged_df['date'].max().date()
    selected_dates = st.sidebar.date_input("Select Date Range", [date_min, date_max], min_value=date_min, max_value=date_max)
    
    selected_sentiment = st.sidebar.multiselect(
        "Market Sentiment", 
        options=merged_df['classification'].unique(), 
        default=merged_df['classification'].unique()
    )
    
    selected_archetype = st.sidebar.multiselect(
        "Trader Archetype",
        options=trader_profile['Archetype Name'].unique(),
        default=trader_profile['Archetype Name'].unique()
    )
    
    # Filter Dataset
    mask = (
        (merged_df['date'].dt.date >= selected_dates[0]) &
        (merged_df['date'].dt.date <= selected_dates[1] if len(selected_dates) > 1 else True) &
        (merged_df['classification'].isin(selected_sentiment)) &
        (merged_df['Archetype Name'].isin(selected_archetype))
    )
    filtered_df = merged_df[mask]

    # --- MAIN DASHBOARD AREA ---
    st.title("NovaTrade AI 🧠 - Strategy & Behavior Hub")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & Metrics", "🧩 Behavioral Clustering", "🧠 Predictive Modeling", "💡 Actionable Insights"])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.header("Portfolio Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total PnL ($)", f"{filtered_df['Closed PnL'].sum():,.2f}")
        with col2:
            st.metric("Total Volume ($)", f"{filtered_df['Size USD'].sum():,.2f}")
        with col3:
            st.metric("Total Trades", f"{len(filtered_df):,}")
        with col4:
            st.metric("Unique Traders", f"{filtered_df['Account'].nunique():,}")
            
        st.subheader("Daily PnL Over Time vs Sentiment")
        # Aggregate daily
        daily_agg = filtered_df.groupby('date')['Closed PnL'].sum().reset_index()
        
        fig = px.bar(daily_agg, x='date', y='Closed PnL', color='Closed PnL',
                     color_continuous_scale=px.colors.diverging.RdYlGn,
                     template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)


    # --- TAB 2: BEHAVIORAL CLUSTERING ---
    with tab2:
        st.header("Trader Behavioral Archetypes")
        st.markdown("We used **K-Means Clustering** to segment traders based on Average Position Size, Trade Frequency, and Win Rate.")
        
        # 3D Scatter
        fig_cluster = px.scatter_3d(
            trader_profile, 
            x='total_trades', 
            y='Size USD', 
            z='win_rate',
            color='Archetype Name',
            hover_data=['Account'],
            template='plotly_dark',
            labels={"total_trades": "Trade Frequency", "Size USD": "Avg Size ($)", "win_rate": "Win Rate"}
        )
        # Tweak plot size
        fig_cluster.update_layout(height=600)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.subheader("Archetype Summary Stats")
        summary_stats = trader_profile.groupby('Archetype Name')[['Size USD', 'total_trades', 'win_rate', 'Closed PnL']].mean().round(2)
        st.dataframe(summary_stats, use_container_width=True)

    # --- TAB 3: PREDICTIVE MODELING ---
    with tab3:
        st.header("Predicting Next-Day Profitability")
        st.markdown("A **Random Forest Classifier** is predicting if a trader will have > $0 PnL the following day based on current and lagged behavioral/sentiment parameters.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Model Performance")
            st.metric("Accuracy Score", f"{accuracy:.2%}")
            st.markdown("Precision (Profitable): **{:.2f}**".format(class_report['1']['precision']))
            st.markdown("Recall (Profitable): **{:.2f}**".format(class_report['1']['recall']))
            st.markdown("F1-Score (Profitable): **{:.2f}**".format(class_report['1']['f1-score']))
        
        with col2:
            st.subheader("Feature Importance")
            fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', 
                             template='plotly_dark', color='Importance', color_continuous_scale='blues')
            st.plotly_chart(fig_imp, use_container_width=True)

    # --- TAB 4: ACTIONABLE INSIGHTS ---
    with tab4:
        st.header("Algorithmic Rules of Thumb")
        st.info("These rules are dynamically generated based on the predictive model's feature importance and clustering traits.")
        
        # Determine top feature
        top_feature = feature_importance.iloc[-1]['Feature']
        second_feature = feature_importance.iloc[-2]['Feature']
        
        insight_1 = f"""
        ### 1. Watch the Lag: `{top_feature}` and `{second_feature}`
        Since **{top_feature}** and **{second_feature}** are the highest drivers for predicting next-day profitability, 
        traders should closely monitor these metrics. For example, if 'value' (Fear/Greed Index numeric value) drops significantly, 
        probability of profitable trades shifts. Modify sizing accordingly before order execution.
        """
        st.markdown(insight_1)
        
        insight_2 = """
        ### 2. Capping the 'High-Frequency Archetype'
        Based on cluster aggregates, the highest trade volume often correlates with lower average win rates.
        *Rule:* If you fall into the 'High-Frequency' cluster, enforce a hard stop at X trades/day to prevent over-trading in choppy, 'Neutral' sentiment regimes.
        """
        st.markdown(insight_2)
        
        insight_3 = """
        ### 3. Scaling Size on Sentiment Flips
        Because current day `Closed PnL` heavily predicts *next* day profitability (momentum effect), losers tend to revenge trade and lose again (mean aversion). 
        *Rule:* If previous day PnL was heavily negative and today is 'Extreme Fear', immediately cut leverage size by 50%.
        """
        st.markdown(insight_3)
