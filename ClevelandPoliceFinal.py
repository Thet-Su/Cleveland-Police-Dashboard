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

# Load the consolidated file
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
df_clean['Street'].replace('', pd.NA, inplace=True)
df_clean = df_clean.dropna(subset=['Street'])

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

nav_options = ['📊 Overview', '📍Crime Hotspots', '📈 Temporal Trends', '🔮 Crime Forecast']
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



if nav == '📊 Overview':

# ========== SECTION: OVERVIEW ==========
    # Title
    st.title("🚔 Cleveland Police Crime Dashboard")
    st.markdown("Understand crime trends, hotspots, and outcomes across Cleveland.")

    st.header("📊 Crime Overview")
    st.markdown("This section summarises Cleveland’s crime trends, including total and average monthly crimes, common locations, dominant offence types, and year-by-year changes, illustrated with supporting charts.")

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

    col_a, col_b, col_c, col_d = st.columns(4)

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
            <h5 style="color:#004085;">{top_street}</h5>
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
    
    with col_d:
        unique_months = filtered_df['Month'].dt.to_period('M').nunique()
        total_crimes = len(filtered_df)

    # Calculate average crimes per month, handling potential division by zero
        if unique_months > 0:
            avg_crimes_per_month = total_crimes / unique_months
        else:
            avg_crimes_per_month = 0
            
        # Round the average to the nearest whole number and convert to an integer
        rounded_avg = int(round(avg_crimes_per_month))

        st.markdown(f"""
        <div style="{card_style}">
            <h6>Avg. Crimes/Month</h6>
            <h5 style="color:#004085;">{rounded_avg:,}</h5>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    crime_type_counts = filtered_df['Crime type'].value_counts().reset_index()
    crime_type_counts.columns = ['Crime Type', 'Count']

    with col1:
        fig1 = px.bar(
            crime_type_counts,
            x='Count',
            y='Crime Type',
            title="Crime Count by Type",
            orientation='h', # Create a horizontal bar chart
            color_discrete_sequence=px.colors.sequential.Blues_r,
            )   
        fig1.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig1)

    with col2:
        year_counts = filtered_df.groupby('Year').size().reset_index(name='Count')
        fig_year = px.bar(
            year_counts,
            x='Year',
            y='Count',
            title="Crime Count by Year",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            )
        fig_year.update_xaxes(type='category')
        st.plotly_chart(fig_year)


# ========== SECTION: LOCATIONS ==========

elif nav == '📍Crime Hotspots':

    st.header("📍 Crime by Street and Map")
 
    loc_df = filtered_df.copy()
    if selected_streets:
            loc_df = loc_df[loc_df['Street'].isin(selected_streets)]

    # --- Top 20 Streets by Crime ---
    st.subheader("🏘️ Top 20 Streets by Crime Count")
    st.markdown("This ranking chart lists the most affected areas/streets, which consistently appear as crime hotspots.")
    street_counts = loc_df['Street'].value_counts().head(20).reset_index()
    street_counts.columns = ['Street', 'Count']
    
    fig2 = px.bar(
    street_counts,
    x='Street',
    y='Count',
    color_discrete_sequence=px.colors.sequential.Blues_r,
    height= 700,
    )
    fig2.update_layout(xaxis_tickangle=-90) # Rotates x-axis labels
    st.plotly_chart(fig2)

    # --- Map ---
    st.subheader("🗺️ Crime Map by Type")
    st.markdown("The geographic plot displays incident locations across Cleveland, with hoverable details for individual crimes.")

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
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("No geolocation data available for selected filter.")


# ========== SECTION: CTRENDS & HEATMAP ==========
elif nav == '📈 Temporal Trends':
    st.header("📈 Crime Trends and Heatmap")
    st.subheader("Monthly Crime Trend")
    st.markdown("The line chart shows how crime incidents fluctuate over time, revealing seasonal peaks and patterns.")

    monthly_trend = filtered_df.groupby(['Month', 'Crime type']).size().reset_index(name='Crime Count')
    pivot = monthly_trend.pivot(index='Month', columns='Crime type', values='Crime Count')

    fig_line = px.line(
            monthly_trend,
            x='Month',
            y='Crime Count',
            color='Crime type',
            labels={'Crime Count': 'Crime Count', 'Month': 'Date'}
        )
        
        # Display the interactive Plotly chart
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Monthly Crime Heatmap")
    st.markdown("This two-dimensional heatmap plots incidents by month and year, providing a compact overview of crime intensity over time.")

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
elif nav == '🔮 Crime Forecast':
    st.header("🔮 Crime Forecast")
    st.markdown("This time-series forecast projects crime levels for the next six months.")


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
    

