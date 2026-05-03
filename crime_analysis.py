import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('crime_data.csv', parse_dates=['DateTime'])

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Feature engineering
df['Month'] = df['DateTime'].dt.month
df['Year'] = df['DateTime'].dt.year
df['DayOfWeek'] = df['DateTime'].dt.dayofweek

# Encode categorical
le_crime = LabelEncoder()
df['CrimeTypeEnc'] = le_crime.fit_transform(df['CrimeType'])
le_loc = LabelEncoder()
df['LocationEnc'] = le_loc.fit_transform(df['Location'])

# Train model
features = ['Hour', 'DayOfWeek', 'Month', 'LocationEnc']
X = df[features]
y = df['CrimeTypeEnc']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders using pickle
import pickle
pickle.dump(model, open('crime_model.pkl', 'wb'))
pickle.dump(le_crime, open('label_encoder_crime.pkl', 'wb'))
pickle.dump(le_loc, open('label_encoder_loc.pkl', 'wb'))

# Calculate risk scores
location_counts = df['Location'].value_counts()
max_count = location_counts.max()
location_risk = location_counts / max_count
df['HighRiskHour'] = df['Hour'].apply(lambda h: 1 if 18 <= h <= 23 else 0)
time_risk_by_loc = df.groupby('Location')['HighRiskHour'].mean()
risk_df = pd.DataFrame({'Location': location_risk.index, 'FrequencyRisk': location_risk.values})
risk_df['TimeRisk'] = risk_df['Location'].map(time_risk_by_loc).fillna(0)
risk_df['OverallRisk'] = (risk_df['FrequencyRisk'] + risk_df['TimeRisk']) / 2
risk_df['RiskLevel'] = pd.cut(risk_df['OverallRisk'], bins=[0, 0.33, 0.66, 1], labels=['Low', 'Medium', 'High'])
risk_df.to_csv('location_risk_scores.csv', index=False)

# Calculate severity scores
severity_base = {
    'Theft': 3, 'Fraud': 4, 'Cyber Crime': 5, 'Burglary': 6,
    'Assault': 7, 'Robbery': 8, 'Kidnapping': 9, 'Homicide': 10
}
df['SeverityBase'] = df['CrimeType'].map(severity_base)
df['TimeMultiplier'] = df['Hour'].apply(lambda h: 1.5 if h >= 22 or h <= 4 else 1.0)
location_severity = {
    'Downtown': 1.3, 'Industrial Area': 1.2, 'Northside': 1.1,
    'Southside': 1.1, 'Eastside': 1.0, 'Westside': 1.0, 'Suburb A': 0.9, 'Suburb B': 0.9
}
df['LocationMultiplier'] = df['Location'].map(location_severity)
df['WomenBoost'] = df['CrimeAgainstWomen'].apply(lambda x: 1 if x == 1 else 0)
df['SeverityScore'] = (df['SeverityBase'] * df['TimeMultiplier'] * df['LocationMultiplier']) + df['WomenBoost']

# Seasonal analysis
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'
df['Season'] = df['DateTime'].dt.month.apply(get_season)

# Police optimization
location_stats = df.groupby('Location').agg({
    'CrimeType': 'count',
    'Latitude': 'mean',
    'Longitude': 'mean',
    'SeverityScore': 'mean'
}).rename(columns={'CrimeType': 'CrimeCount'})
location_stats['PatrolPriority'] = (location_stats['CrimeCount'] * 0.6) + (location_stats['SeverityScore'] * 0.4)
location_stats = location_stats.sort_values('PatrolPriority', ascending=False)

# K-means for patrol points
high_crime_coords = df[df['Location'].isin(location_stats.head(5).index)][['Latitude', 'Longitude']].values
if len(high_crime_coords) >= 3:
    kmeans = KMeans(n_clusters=min(3, len(high_crime_coords)), random_state=42, n_init=10)
    kmeans.fit(high_crime_coords)
    patrol_points = kmeans.cluster_centers_
    pd.DataFrame(patrol_points, columns=['Latitude', 'Longitude']).to_csv('patrol_points.csv', index=False)

# Save processed data
df.to_csv('processed_crime_data.csv', index=False)

print("✅ Analysis complete! All data and models saved.")