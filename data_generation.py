import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Create a folder for plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
num_records = 5000
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)

crime_types = ['Theft', 'Assault', 'Burglary', 'Robbery', 'Homicide', 'Cyber Crime', 'Fraud', 'Kidnapping']
locations = ['Downtown', 'Northside', 'Southside', 'Eastside', 'Westside', 'Suburb A', 'Suburb B', 'Industrial Area']
lat_center, lon_center = 28.6139, 77.2090
lat_range, lon_range = 0.05, 0.05

crime_keywords = {
    'Theft': ['stole', 'wallet', 'missing', 'pickpocket', 'purse'],
    'Assault': ['punched', 'kicked', 'attack', 'injury', 'beaten'],
    'Burglary': ['broke', 'entered', 'forced', 'window', 'door'],
    'Robbery': ['gun', 'threat', 'demanded', 'forcibly', 'mugged'],
    'Homicide': ['dead', 'murder', 'stabbed', 'shot', 'killed'],
    'Cyber Crime': ['hacked', 'phishing', 'ransomware', 'identity theft'],
    'Fraud': ['scam', 'fake', 'cheated', 'forgery'],
    'Kidnapping': ['abducted', 'missing child', 'ransom', 'held captive']
}

def generate_description(crime_type):
    keywords = crime_keywords[crime_type]
    return f"{random.choice(keywords)} incident reported. {random.choice(['Evidence collected', 'Witness present', 'CCTV available', 'No leads yet'])}"

data = []
for _ in range(num_records):
    days_diff = (end_date - start_date).days
    random_days = random.randint(0, days_diff)
    crime_date = start_date + timedelta(days=random_days)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    crime_datetime = crime_date.replace(hour=hour, minute=minute)
    
    crime = random.choice(crime_types)
    loc = random.choice(locations)
    lat = lat_center + np.random.normal(0, lat_range/3)
    lon = lon_center + np.random.normal(0, lon_range/3)
    
    victim_gender = random.choice(['Male', 'Female', 'Unknown'])
    crime_against_women = 1 if (victim_gender == 'Female' and crime in ['Assault', 'Kidnapping', 'Robbery']) else 0
    if random.random() < 0.1:
        crime_against_women = 1
    
    description = generate_description(crime)
    
    data.append([crime_datetime, hour, crime_datetime.weekday(), crime, loc, lat, lon,
                 description, victim_gender, crime_against_women])

df = pd.DataFrame(data, columns=['DateTime', 'Hour', 'DayOfWeek', 'CrimeType', 'Location', 'Latitude', 'Longitude',
                                 'Description', 'VictimGender', 'CrimeAgainstWomen'])
df.to_csv('crime_data.csv', index=False)
print(f"✅ Generated {num_records} records and saved to crime_data.csv")