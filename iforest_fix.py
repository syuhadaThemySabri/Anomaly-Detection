import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(file_path, contamination=0.02):
    """
    Detect anomalies in turbine performance data using Isolation Forest.
    
    Parameters:
    file_path (str): Path to the CSV file containing the dataset.
    contamination (float): The expected proportion of outliers in the data (default: 0.02).
    
    Returns:
    pandas.DataFrame: DataFrame with processed data, anomaly labels, and anomaly scores.
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Convert timestamp and create time features
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], utc=True)
    data['Hour'] = data['Timestamp'].dt.hour
    data['Day'] = data['Timestamp'].dt.day
    data['Month'] = data['Timestamp'].dt.month
    data['Weekday'] = data['Timestamp'].dt.weekday

    # List of feature columns
    feature_columns = ['Ngp', 'Npt', 'HPC_ASV_Command', 'HPC_ASV_Position',
                       'HPC_Surge_Margin', 'HPC_ASC_Flow_DP', 'HPC_Suction_Press',
                       'HPC_Discharge_Press']

    # Convert features to numeric
    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Calculate rolling statistics
    window_size = 24  # 24-hour window
    for col in feature_columns:
        data[f'{col}_rolling_mean'] = data[col].rolling(window=window_size).mean()
        data[f'{col}_rolling_std'] = data[col].rolling(window=window_size).std()

    # Drop rows with NaN values
    data = data.dropna()

    # Select features for anomaly detection
    feature_cols = [col for col in data.columns if any(x in col for x in ['_rolling_mean', '_rolling_std'])]

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[feature_cols])

    # Train Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )

    # Fit and predict
    data['anomaly'] = iso_forest.fit_predict(scaled_features)
    data['anomaly_score'] = iso_forest.score_samples(scaled_features)

    return data