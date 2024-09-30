import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read and preprocess data as in original code
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
weather_df = pd.read_csv('weather_data_nyc_centralpark_2016.csv')

# Replace 'T' with a small value for precipitation
weather_df.replace('T', 0.001, inplace=True)

train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime']).dt.date
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime']).dt.date
weather_df['date'] = pd.to_datetime(weather_df['date'], format='%d-%m-%Y').dt.date

train_merged = pd.merge(train_df, weather_df, how='left', left_on='pickup_datetime', right_on='date')
test_merged = pd.merge(test_df, weather_df, how='left', left_on='pickup_datetime', right_on='date')

# Function to calculate haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

train_merged['distance_km'] = haversine(train_merged['pickup_latitude'], train_merged['pickup_longitude'],
                                        train_merged['dropoff_latitude'], train_merged['dropoff_longitude'])
test_merged['distance_km'] = haversine(test_merged['pickup_latitude'], test_merged['pickup_longitude'],
                                       test_merged['dropoff_latitude'], test_merged['dropoff_longitude'])

train_merged['pickup_hour'] = pd.to_datetime(train_merged['pickup_datetime']).dt.hour
train_merged['rush_hour'] = ((train_merged['pickup_hour'] >= 7) & (train_merged['pickup_hour'] <= 9)) | \
                            ((train_merged['pickup_hour'] >= 16) & (train_merged['pickup_hour'] <= 19))
train_merged['rush_hour'] = train_merged['rush_hour'].astype(int)

train_merged['pickup_hour'] = pd.to_datetime(train_merged['pickup_datetime']).dt.hour
train_merged['day_of_week'] = pd.to_datetime(train_merged['pickup_datetime']).dt.weekday
train_merged['month'] = pd.to_datetime(train_merged['pickup_datetime']).dt.month

# Select features and target
features = ['passenger_count', 'distance_km', 'rush_hour', 'average temperature', 'precipitation', 'pickup_hour', 'day_of_week', 'month']
X_train = train_merged[features].values
y_train = train_merged['trip_duration'].values

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

y_train = np.log(y_train)
y_test = np.log(y_test)

# Define the model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(len(features), 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Dropout to reduce overfitting
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training loop
epochs = 100
train_losses = []
eval_losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_pred_train = model.forward(X_train)
    train_loss = criterion(y_pred_train, y_train)
    train_losses.append(train_loss.detach().numpy())
    
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model.forward(X_test)
        val_loss = criterion(y_pred, y_test)
    eval_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {train_loss.item():.4f}')

# Evaluation with clipped predictions to avoid overflow issues
def evaluate_model(actuals, predictions):
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')
    return mae, rmse, r2

y_pred_exp = y_pred.detach().numpy().flatten()
y_test_exp = np.exp(y_test.numpy().flatten())

print("Evaluation on Test Set:")
evaluate_model(y_test_exp, y_pred_exp)

# Plotting training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(eval_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
