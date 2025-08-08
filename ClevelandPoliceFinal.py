import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
from datetime import datetime
import streamlit.components.v1 as components

#----------------
# Data Loading
#----------------

# Load the saved file, renamed and converted to xlsx 
df_clean = pd.read_csv('cleveland_final.csv')

#See the data type of each column in your DataFrame
print(df_clean.dtypes)

#----------------
# Data Cleaning and Preparation
#----------------

# 1. Fill blanks in 'Last outcome category' with "No outcome yet"
df_clean['Last outcome category'] = df_clean['Last outcome category'].replace('', pd.NA)
df_clean['Last outcome category'] = df_clean['Last outcome category'].fillna('No outcome yet')

# 2. Convert 'Month' column (format 'YYYY-MM') to datetime (using first day of month)
df_clean['Month'] = pd.to_datetime(df_clean['Month'], format='%Y-%m')

# 3. Convert "Crime type" into categories and counts
categories = df_clean['Crime type'].astype('category').cat.categories
counts = df_clean['Crime type'].value_counts().sort_index()

# 4. Drop unnecessary columns
columns_to_drop = ['Crime ID', 'Reported by', 'Falls within', 'Context']
df_clean = df_clean.drop(columns=columns_to_drop)

# 5. Create 'Street' column by removing "On or Near" from 'Location'
df_clean['Street'] = df_clean['Location'].str.replace('On or near', '').str.strip()


# 6. Extract Month Name and Year
df_clean['Month_Name'] = df_clean['Month'].dt.strftime('%B')
df_clean['Year'] = df_clean['Month'].dt.year

print(df_clean.dtypes)
print(df_clean.head())

#----------------
# Dashboard Development
#----------------

# Streamlit Config
st.set_page_config(page_title="Cleveland Crime Dashboard", layout="wide")

# ---- Horizontal Navigation ----
st.markdown(
    """
    <style>
        .nav-container {
            display: flex;
            justify-content: center;
            gap: 2rem;
            padding: 1rem 0;
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        .nav-item {
            font-weight: bold;
            color: #004085;
            text-decoration: none;
        }
        .nav-item-selected {
            color: #fff !important;
            background-color: #004085;
            padding: 0.5rem 1rem;
            border-radius: 999px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

nav_options = ['Overview', 'Locations', 'Correlation', 'Trends', 'Forecast', 'Insights']
nav = st.radio(
    "", nav_options, horizontal=True, label_visibility="collapsed"
)


# Sidebar with Logo and Filters
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/1/15/Cleveland_Police_logo.svg/1200px-Cleveland_Police_logo.svg.png", width=120)
    st.title(":mag: Filter Options")

  # Date Range Filter
    min_date, max_date = df_clean['Month'].min(), df_clean['Month'].max()
    selected_dates = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    # Other Filters
    available_crime_types = sorted(df_clean['Crime type'].dropna().unique())
    available_outcomes = sorted(df_clean['Last outcome category'].dropna().unique())
    available_streets = sorted(df_clean['Street'].dropna().unique())

    selected_crime_types = st.multiselect("Crime Types", available_crime_types)
    selected_outcomes = st.multiselect("Outcome Categories", available_outcomes)
    selected_streets = st.multiselect("Streets", available_streets)

# Filtered Data
filtered_df = df_clean[
    (df_clean['Month'] >= pd.to_datetime(selected_dates[0])) &
    (df_clean['Month'] <= pd.to_datetime(selected_dates[1]))
]

if selected_crime_types:
    filtered_df = filtered_df[filtered_df['Crime type'].isin(selected_crime_types)]

if selected_outcomes:
    filtered_df = filtered_df[filtered_df['Last outcome category'].isin(selected_outcomes)]

if selected_streets:
    filtered_df = filtered_df[filtered_df['Street'].isin(selected_streets)]



# Title
st.title("🚔 Cleveland Police Crime Dashboard")
st.markdown("Understand crime trends, hotspots, and outcomes across Cleveland.")

if nav == 'Overview':

# ========== SECTION: OVERVIEW ==========
    st.header("📊 Crime Overview")

    card_style = """
    border:2px solid #DEE2E6;
    border-radius:8px;
    padding:1rem;
    text-align:center;
    height:130px;
    display:flex;
    flex-direction:column;
    justify-content:center;
"""

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f"""
        <div style="{card_style}">
            <h6>Total Crimes</h6>
            <h5 style="color:#004085;">{len(filtered_df):,}</h5>
        </div>
    """, unsafe_allow_html=True)

    with col_b:
        top_street = filtered_df['Street'].mode().iloc[0] if not filtered_df['Street'].isna().all() else "N/A"
        st.markdown(f"""
        <div style="{card_style}">
            <h6>Most Frequent Crime Spot</h6>
            <h6 style="color:#004085;">{top_street}</h6>
        </div>
    """, unsafe_allow_html=True)

    with col_c:
        top_crime = filtered_df['Crime type'].mode().iloc[0] if not filtered_df['Crime type'].isna().all() else "N/A"
        st.markdown(f"""
        <div style="{card_style}">
            <h6>Most Frequent Crime Type</h6>
            <h5 style="color:#004085;">{top_crime}</h5>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.countplot(
            data=filtered_df,
            y='Crime type',
            order=filtered_df['Crime type'].value_counts().index,
            palette='Blues_r',
            ax=ax1
            )
        ax1.set_title("Crime Count by Type")
        st.pyplot(fig1)

    with col2:
        fig_year, ax_year = plt.subplots(figsize=(10, 4))
        filtered_df.groupby('Year').size().plot(kind='bar', color='steelblue', ax=ax_year)
        ax_year.set_title("Crime Count by Year")
        ax_year.set_ylabel("Number of Crimes")
        st.pyplot(fig_year)


# ========== SECTION: LOCATIONS ==========

elif nav == 'Locations':

    st.header("📍 Crime by Street and Map")
    st.markdown("**Explore:** Are some streets repeatedly hotspots for crime?")
    loc_df = filtered_df.copy()
    if selected_streets:
            loc_df = loc_df[loc_df['Street'].isin(selected_streets)]

    col1, col2 = st.columns(2)
    with col1:
        street_counts = loc_df['Street'].value_counts().head(20).reset_index()
        street_counts.columns = ['Street', 'Count']
        fig2, ax2 = plt.subplots(figsize=(6, 8))
        sns.barplot(data=street_counts, y='Count', x='Street', palette='Blues_r', ax=ax2)
        ax2.set_title("Top 20 Streets by Crime Count")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)
        

    with col2:
        st.caption("🧭 Tip: Click on a legend item to toggle a crime type on/off.")
        map_df = loc_df.dropna(subset=['Latitude', 'Longitude'])
        if not map_df.empty:
            fig_map = px.scatter_mapbox(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color="Crime type",
                hover_data=["LSOA name", "Street", "Last outcome category"],
                zoom=10,
                height=650,
                mapbox_style="carto-positron"
                )
            st.plotly_chart(fig_map)
        else:
                    st.warning("No geolocation data available for selected filter.")


# ========== SECTION: CORRELATION ==========
elif nav == 'Correlation':
    st.header("🧮 Crime Type Correlation Matrix")
    st.markdown("**Question:** What types of crime dominate in Cleveland? How has it changed annually?")

    pivot_corr = filtered_df.groupby(['Month', 'Crime type']).size().unstack(fill_value=0)
    corr_matrix = pivot_corr.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Correlation Between Crime Types Over Time")
    st.pyplot(fig_corr)

# Additional sections like Trends, Outcomes, etc. continue below...


# ========== SECTION: CTRENDS & HEATMAP ==========
elif nav == 'Trends':
    st.header("📈 Crime Trends and Heatmap")
    col_trend, col_heat = st.columns(2)

    with col_trend:
        st.subheader("Monthly Crime Trend")
        monthly_trend = filtered_df.groupby(['Month', 'Crime type']).size().reset_index(name='Crime Count')
        pivot = monthly_trend.pivot(index='Month', columns='Crime type', values='Crime Count')

        fig_line, ax_line = plt.subplots(figsize=(10, 5))
        pivot.plot(ax=ax_line)
        ax_line.set_title("Detailed Crime Trend")
        ax_line.set_ylabel("Crime Count")
        ax_line.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig_line.tight_layout()
        st.pyplot(fig_line)

    with col_heat:
        st.subheader("Monthly Crime Heatmap")
        heat_df = filtered_df.copy()
        heat_df['Month'] = heat_df['Month'].dt.strftime('%B')
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
        month_year = heat_df.groupby(['Year', 'Month_Name'], observed=False).size().unstack().T
        month_year = month_year.reindex(month_order)
        fig_heat, ax_heat = plt.subplots(figsize=(10, 5))
        sns.heatmap(month_year, annot=True, fmt=".0f", cmap='YlOrRd', ax=ax_heat)
        ax_heat.set_title("Monthly Crime Volume")
        st.pyplot(fig_heat)

# ========== SECTION: CRIME FORECAST ==========
elif nav == 'Forecast':
    st.header("Crime Forecast")

    try:
        from prophet import Prophet
        from prophet.plot import plot_plotly
        import warnings
        warnings.filterwarnings("ignore")
    
    # Prepare time series data
        ts_df = filtered_df.groupby('Month').size().reset_index(name='y')
        ts_df.rename(columns={'Month': 'ds'}, inplace=True)
    
    # Fit model
        model = Prophet()
        model.fit(ts_df)
    
    # Future dataframe
        future = model.make_future_dataframe(periods=6, freq='MS')
        forecast = model.predict(future)
    
    # Plot forecast
        st.subheader("📈 Crime Forecast for Next 6 Months")
        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    except ImportError:
        st.error("`prophet` package not found. Please install it using `pip install prophet`.")

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
    
    # ========== INSIGHT PLACEHOLDERS ==========
st.markdown("---\n### 🔍 Summary Insights")
st.info("""**Street Hotspots:** Carlton Avenue & Stockton area top the crime count.

**Seasonality:** July–August show visible spikes. Investigate why.

**Type Dominance:** 'Violence and sexual offences' remain most frequent crime type each year.""")


