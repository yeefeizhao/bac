import pandas as pd
import folium

# Load your CSV data
df = pd.read_csv('train.csv')

# Create a map centered around NYC (lat, long: 40.7128, -74.0060)
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

# Plot pickup points in blue
for lat, lon in zip(df['pickup_latitude'], df['pickup_longitude']):
    folium.CircleMarker([lat, lon], radius=2, color='blue', fill=True).add_to(nyc_map)

# Plot dropoff points in red
for lat, lon in zip(df['dropoff_latitude'], df['dropoff_longitude']):
    folium.CircleMarker([lat, lon], radius=2, color='red', fill=True).add_to(nyc_map)

# Save and display the map
nyc_map.save('nyc_pickup_dropoff_map.html')
