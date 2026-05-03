import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crime Analysis System",
    page_icon="🚔",
    layout="wide"
)

st.title("🚔 Crime Data Analysis & Prediction System")

# Check if required files exist
required_files = ['processed_crime_data.csv', 'location_risk_scores.csv', 'crime_model.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"""
    ❌ **Missing required files!**
    
    Please run these commands:
    
    1. `python data_generation.py`
    2. `python crime_analysis.py`
    
    Missing: {', '.join(missing_files)}
    """)
    st.stop()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('processed_crime_data.csv', parse_dates=['DateTime'])
    risk_df = pd.read_csv('location_risk_scores.csv')
    return df, risk_df

@st.cache_resource
def load_model():
    model = pickle.load(open('crime_model.pkl', 'rb'))
    le_crime = pickle.load(open('label_encoder_crime.pkl', 'rb'))
    le_loc = pickle.load(open('label_encoder_loc.pkl', 'rb'))
    return model, le_crime, le_loc

df, risk_df = load_data()
model, le_crime, le_loc = load_model()

st.success(f"✅ Loaded {len(df)} crime records")

st.markdown("---")

# ==================== SIDEBAR FILTERS ====================
with st.sidebar:
    st.markdown("## 🔍 Data Filters")
    st.markdown("*Leave as 'All' to see full data*")
    
    # Date range - with better defaults
    min_date = df['DateTime'].min().date()
    max_date = df['DateTime'].max().date()
    date_range = st.date_input("📅 Date Range", value=(min_date, max_date))
    
    # Hour range - with better defaults
    hour_range = st.slider("🕐 Hour Range", 0, 23, (0, 23), help="Select specific hours")
    
    # Day of week - with All as default
    days = ['All', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    selected_day = st.selectbox("📅 Day of Week", days, index=0)
    
    # Location - with All as default
    locations = ['All'] + sorted(df['Location'].unique().tolist())
    selected_location = st.selectbox("📍 Location", locations, index=0)
    
    # Crime type - with All as default
    crime_types = ['All'] + sorted(df['CrimeType'].unique().tolist())
    selected_crime = st.selectbox("📊 Crime Type", crime_types, index=0)
    
    # Reset button
    col1, col2 = st.columns(2)
    with col1:
        apply_filters = st.button("🔄 Apply Filters", type="primary", use_container_width=True)
    with col2:
        reset_filters = st.button("🗑️ Reset All", use_container_width=True)

# Reset functionality
if reset_filters:
    st.session_state.clear()
    st.rerun()

# ==================== FILTER FUNCTION ====================
def filter_data(df, date_range, hour_range, day, location, crime):
    filtered = df.copy()
    
    # Apply date filter
    if len(date_range) == 2:
        filtered = filtered[
            (filtered['DateTime'].dt.date >= date_range[0]) & 
            (filtered['DateTime'].dt.date <= date_range[1])
        ]
    
    # Apply hour filter
    if hour_range[0] != 0 or hour_range[1] != 23:
        filtered = filtered[
            (filtered['Hour'] >= hour_range[0]) & 
            (filtered['Hour'] <= hour_range[1])
        ]
    
    # Apply day filter
    if day != 'All':
        day_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
        filtered = filtered[filtered['DayOfWeek'] == day_map[day]]
    
    # Apply location filter
    if location != 'All':
        filtered = filtered[filtered['Location'] == location]
    
    # Apply crime filter
    if crime != 'All':
        filtered = filtered[filtered['CrimeType'] == crime]
    
    return filtered

# Initialize session state
if 'filtered_df' not in st.session_state or apply_filters:
    st.session_state.filtered_df = filter_data(df, date_range, hour_range, selected_day, selected_location, selected_crime)

filtered_df = st.session_state.filtered_df

# If no data after filtering, show warning and suggest reset
if len(filtered_df) == 0:
    st.warning("⚠️ No data matches the selected filters!")
    st.info("💡 **Suggestions:**\n"
            "- Click 'Reset All' to see full data\n"
            "- Select fewer filters\n"
            "- Expand your date range\n"
            "- Choose 'All' for location or crime type")
    
    # Show quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📅 Show Last Month Only", use_container_width=True):
            last_month = df[df['DateTime'] >= df['DateTime'].max() - pd.Timedelta(days=30)]
            st.session_state.filtered_df = last_month
            st.rerun()
    with col2:
        if st.button("🌙 Show Night Crimes Only", use_container_width=True):
            night_crimes = df[df['Hour'].between(22, 24) | df['Hour'].between(0, 4)]
            st.session_state.filtered_df = night_crimes
            st.rerun()
    with col3:
        if st.button("👩 Show Women-Related Crimes", use_container_width=True):
            women_crimes = df[df['CrimeAgainstWomen'] == 1]
            st.session_state.filtered_df = women_crimes
            st.rerun()
    st.stop()

# Calculate percentage of total
pct_of_total = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0

# Show filter summary
st.info(f"📊 Showing **{len(filtered_df)}** crimes ({pct_of_total:.1f}% of total data)")

# Show applied filters
active_filters = []
if selected_location != 'All':
    active_filters.append(f"📍 {selected_location}")
if selected_crime != 'All':
    active_filters.append(f"📊 {selected_crime}")
if selected_day != 'All':
    active_filters.append(f"📅 {selected_day}")
if hour_range[0] != 0 or hour_range[1] != 23:
    active_filters.append(f"🕐 {hour_range[0]:02d}:00 - {hour_range[1]:02d}:00")

if active_filters:
    st.caption(f"**Active filters:** {' | '.join(active_filters)}")

st.markdown("---")

# ==================== TOP METRICS ====================
st.header("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Crimes", f"{len(filtered_df):,}")
with col2:
    most_common = filtered_df['CrimeType'].mode()[0] if len(filtered_df) > 0 else "N/A"
    st.metric("Most Common Crime", most_common)
with col3:
    peak_hour = filtered_df.groupby('Hour').size().idxmax() if len(filtered_df) > 0 else 0
    st.metric("Peak Hour", f"{peak_hour}:00")
with col4:
    women_crimes = filtered_df['CrimeAgainstWomen'].sum()
    st.metric("Crimes vs Women", f"{women_crimes:,}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    busiest_season = filtered_df['Season'].mode()[0] if len(filtered_df) > 0 else "N/A"
    st.metric("Busiest Season", busiest_season)
with col2:
    avg_severity = filtered_df['SeverityScore'].mean() if len(filtered_df) > 0 else 0
    st.metric("Avg Severity", f"{avg_severity:.1f}/10")
with col3:
    night_crimes = len(filtered_df[filtered_df['Hour'].between(22, 24) | filtered_df['Hour'].between(0, 4)])
    night_pct = (night_crimes / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Night Crimes", f"{night_pct:.1f}%")
with col4:
    st.metric("Active Locations", filtered_df['Location'].nunique())

st.markdown("---")

# ==================== PREDICTION ====================
st.header("🎯 Real-Time Crime Prediction")

col1, col2, col3 = st.columns(3)
with col1:
    pred_loc = st.selectbox("📍 Location", df['Location'].unique())
with col2:
    pred_hour = st.slider("🕐 Hour", 0, 23, 12)
with col3:
    pred_day = st.selectbox("📅 Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

if st.button("🔮 Predict Crime", type="primary", use_container_width=True):
    try:
        day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
        loc_enc = le_loc.transform([pred_loc])[0] if pred_loc in le_loc.classes_ else 0
        
        input_data = pd.DataFrame([[pred_hour, day_map[pred_day], df['Month'].mode()[0], loc_enc]],
                                  columns=['Hour', 'DayOfWeek', 'Month', 'LocationEnc'])
        
        pred = model.predict(input_data)[0]
        crime_pred = le_crime.inverse_transform([pred])[0]
        proba = model.predict_proba(input_data)[0]
        confidence = max(proba) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"### 🚨 {crime_pred}")
        with col2:
            st.info(f"Confidence: {confidence:.1f}%")
    except:
        st.success(f"### 🚨 {filtered_df['CrimeType'].mode()[0] if len(filtered_df) > 0 else 'Theft'}")
        st.info("Using historical trend analysis")

st.markdown("---")

# ==================== CHARTS SECTION ====================
st.header("📈 Crime Pattern Analysis")

# Row 1: Crime Distribution & Hourly Pattern
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Crime Type Distribution")
    crime_counts = filtered_df['CrimeType'].value_counts().reset_index()
    crime_counts.columns = ['Crime Type', 'Count']
    if len(crime_counts) > 0:
        fig = px.bar(crime_counts, x='Count', y='Crime Type', orientation='h',
                     color='Count', color_continuous_scale='Reds',
                     title=f"Total: {len(filtered_df)} crimes")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available - adjust filters")

with col2:
    st.subheader("2️⃣ Hourly Crime Pattern")
    hourly = filtered_df.groupby('Hour').size().reset_index()
    hourly.columns = ['Hour', 'Count']
    if len(hourly) > 0:
        fig = px.line(hourly, x='Hour', y='Count', markers=True,
                      title="Crimes Throughout the Day")
        fig.update_traces(line=dict(color='darkred', width=2))
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available - adjust filters")

# Row 2: Day of Week & Seasonal
col1, col2 = st.columns(2)

with col1:
    st.subheader("3️⃣ Day of Week Pattern")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    filtered_df['DayName'] = filtered_df['DateTime'].dt.day_name()
    weekly = filtered_df.groupby('DayName').size().reset_index()
    weekly.columns = ['Day', 'Count']
    if len(weekly) > 0:
        fig = px.bar(weekly, x='Day', y='Count', color='Count',
                     color_continuous_scale='Reds', category_orders={'Day': days_order})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available - adjust filters")

with col2:
    st.subheader("4️⃣ Seasonal Distribution")
    seasonal = filtered_df['Season'].value_counts().reset_index()
    seasonal.columns = ['Season', 'Count']
    if len(seasonal) > 0:
        season_colors = {'Winter': '#3498db', 'Spring': '#2ecc71', 'Summer': '#f1c40f', 'Fall': '#e67e22'}
        fig = px.pie(seasonal, values='Count', names='Season', hole=0.3,
                     color='Season', color_discrete_map=season_colors)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available - adjust filters")

st.markdown("---")

# ==================== RISK & SEVERITY ====================
st.header("⚠️ Risk & Severity Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("5️⃣ High-Risk Locations")
    risk_by_loc = filtered_df.groupby('Location')['SeverityScore'].mean().sort_values(ascending=False).head(8).reset_index()
    risk_by_loc.columns = ['Location', 'Risk Score']
    if len(risk_by_loc) > 0:
        fig = px.bar(risk_by_loc, x='Risk Score', y='Location', orientation='h',
                     color='Risk Score', color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available - adjust filters")

with col2:
    st.subheader("6️⃣ Severity by Crime Type")
    severity = filtered_df.groupby('CrimeType')['SeverityScore'].mean().sort_values(ascending=True).reset_index()
    severity.columns = ['Crime Type', 'Severity']
    if len(severity) > 0:
        fig = px.bar(severity, x='Severity', y='Crime Type', orientation='h',
                     color='Severity', color_continuous_scale='Reds')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available - adjust filters")

st.markdown("---")

# ==================== POLICE RESOURCE (FIXED) ====================
st.header("🚓 Police Resource Optimization")

location_counts = filtered_df['Location'].value_counts()
total_crimes = len(filtered_df)

if total_crimes > 0 and len(location_counts) > 0:
    # Create allocation data
    allocation_data = []
    
    for loc, count in location_counts.head(6).items():
        percentage = (count / total_crimes) * 100
        allocation_data.append({
            'Location': loc,
            'Crimes': count,
            'Allocation %': round(percentage, 1)
        })
    
    # Add "Others" if needed
    if len(location_counts) > 6:
        others_count = location_counts.iloc[6:].sum()
        others_pct = (others_count / total_crimes) * 100
        allocation_data.append({
            'Location': 'Other Locations',
            'Crimes': others_count,
            'Allocation %': round(others_pct, 1)
        })
    
    alloc_df = pd.DataFrame(allocation_data)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = px.pie(alloc_df, values='Allocation %', names='Location',
                     title=f"Patrol Resource Allocation\nBased on {total_crimes} crimes",
                     hole=0.3,
                     color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Deployment Strategy")
        for _, row in alloc_df.iterrows():
            if row['Allocation %'] > 30:
                st.error(f"🚨 **CRITICAL**: {row['Location']} - {row['Allocation %']:.0f}%")
            elif row['Allocation %'] > 15:
                st.warning(f"⚠️ **HIGH**: {row['Location']} - {row['Allocation %']:.0f}%")
            elif row['Allocation %'] > 5:
                st.info(f"ℹ️ **MEDIUM**: {row['Location']} - {row['Allocation %']:.0f}%")
            else:
                st.success(f"✅ **LOW**: {row['Location']} - {row['Allocation %']:.0f}%")
            st.progress(row['Allocation %'] / 100)
else:
    st.info("Not enough data for resource allocation - try resetting filters")

st.markdown("---")

# ==================== TREND FORECASTING ====================
st.header("📈 Crime Trend Forecasting")

monthly_trend = filtered_df.groupby(filtered_df['DateTime'].dt.to_period('M')).size().reset_index()
monthly_trend.columns = ['Month', 'Crime Count']
monthly_trend['Month'] = monthly_trend['Month'].astype(str)

if len(monthly_trend) >= 3:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(monthly_trend, x='Month', y='Crime Count', markers=True,
                      title='Monthly Crime Trend')
        fig.update_traces(line=dict(color='darkred', width=2))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        recent_avg = monthly_trend['Crime Count'].tail(3).mean()
        overall_avg = monthly_trend['Crime Count'].mean()
        trend_pct = ((recent_avg - overall_avg) / overall_avg * 100) if overall_avg > 0 else 0
        
        st.subheader("Trend Analysis")
        if trend_pct > 10:
            st.warning(f"📈 Crime INCREASING ({trend_pct:.1f}% above average)")
        elif trend_pct < -10:
            st.success(f"📉 Crime DECREASING ({abs(trend_pct):.1f}% below average)")
        else:
            st.info(f"📊 Crime STABLE ({trend_pct:.1f}% change)")
        
        st.metric("Recent Average", f"{recent_avg:.0f} crimes/month")
        st.metric("Overall Average", f"{overall_avg:.0f} crimes/month")
else:
    st.info("Need at least 3 months of data for trend analysis")

st.markdown("---")

# ==================== TOP LOCATIONS ====================
st.subheader("📍 Top 10 High-Crime Locations")

top10 = filtered_df['Location'].value_counts().head(10).reset_index()
top10.columns = ['Location', 'Crime Count']

if len(top10) > 0:
    fig = px.bar(top10, x='Crime Count', y='Location', orientation='h',
                 color='Crime Count', color_continuous_scale='Reds',
                 title='Areas Requiring Immediate Attention')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==================== SMART RECOMMENDATIONS ====================
st.header("💡 Smart Recommendations")

if len(filtered_df) > 0:
    top_crime = filtered_df['CrimeType'].mode()[0]
    top_crime_pct = (filtered_df['CrimeType'].value_counts().iloc[0] / len(filtered_df)) * 100
    
    top_loc = filtered_df['Location'].value_counts().index[0]
    top_loc_pct = (filtered_df['Location'].value_counts().iloc[0] / len(filtered_df)) * 100
    
    peak_hour_val = filtered_df.groupby('Hour').size().idxmax()
    peak_hour_pct = (filtered_df.groupby('Hour').size().max() / len(filtered_df)) * 100
    
    night_pct_val = (len(filtered_df[filtered_df['Hour'].between(22, 24) | filtered_df['Hour'].between(0, 4)]) / len(filtered_df)) * 100
    
    women_pct_val = (filtered_df['CrimeAgainstWomen'].sum() / len(filtered_df)) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Immediate Actions")
        
        actions = [
            f"🔴 Focus on **{top_crime}** ({top_crime_pct:.1f}% of crimes)",
            f"🔴 Peak hour at {peak_hour_val}:00 ({peak_hour_pct:.1f}% of incidents)",
            f"📍 {top_loc} needs {top_loc_pct:.1f}% of patrol resources"
        ]
        
        if night_pct_val > 30:
            actions.append(f"🌙 Night patrol needed ({night_pct_val:.1f}% crimes at night)")
        else:
            actions.append(f"☀️ Daytime focus ({100-night_pct_val:.1f}% during day)")
        
        if women_pct_val > 15:
            actions.append(f"👩 Women safety unit required ({women_pct_val:.1f}% vs women)")
        
        for action in actions:
            st.markdown(f"- {action}")
    
    with col2:
        st.markdown("### 📅 Strategic Planning")
        
        busiest_season = filtered_df['Season'].mode()[0]
        season_pct_val = (filtered_df[filtered_df['Season'] == busiest_season].shape[0] / len(filtered_df)) * 100
        
        strategies = [
            f"🌤️ Prepare for {busiest_season} ({season_pct_val:.1f}% of crimes)",
            f"🗺️ Focus on top 3 locations",
        ]
        
        unique_crimes = filtered_df['CrimeType'].nunique()
        if unique_crimes > 4:
            strategies.append(f"📊 Multi-crime training needed ({unique_crimes} types)")
        
        avg_severity_val = filtered_df['SeverityScore'].mean()
        if avg_severity_val > 7:
            strategies.append(f"⚠️ High severity response team needed")
        
        for strategy in strategies:
            st.markdown(f"- {strategy}")

st.markdown("---")

# ==================== DATA TABLE ====================
st.header("📋 Recent Crime Incidents")

recent = filtered_df.nlargest(20, 'DateTime')[['DateTime', 'CrimeType', 'Location', 'Hour', 'SeverityScore', 'CrimeAgainstWomen']]
recent['DateTime'] = recent['DateTime'].dt.strftime('%Y-%m-%d %H:%M')
recent = recent.rename(columns={
    'DateTime': 'Date & Time',
    'CrimeType': 'Crime Type',
    'Location': 'Location',
    'Hour': 'Hour',
    'SeverityScore': 'Severity',
    'CrimeAgainstWomen': 'Women Related'
})
st.dataframe(recent, use_container_width=True)

st.markdown("---")

# ==================== FOOTER ====================
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
    <p>🚔 <b>Crime Data Analysis & Prediction System</b></p>
    <p>📊 8 Interactive Charts | 🎯 Real-time Prediction | 🚓 Smart Resource Allocation | 💡 Dynamic Recommendations</p>
    <p>💡 <b>Tip:</b> Use 'Reset All' if you see no data - or try the quick action buttons above</p>
</div>
""", unsafe_allow_html=True)