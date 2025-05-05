import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def detect_anomalies(file_path):
    """
    Detect anomalies in turbine performance data using One-Class SVM.
    
    Parameters:
    file_path (str): Path to the CSV file containing the dataset.
    
    Returns:
    pandas.DataFrame: DataFrame with aggregated data, anomaly labels, and anomaly scores.
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Parsing timestamp
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])

    # List of feature columns
    feature_columns = ['Ngp', 'Npt', 'HPC_ASV_Command', 'HPC_ASV_Position', 'HPC_Surge_Margin',
                       'HPC_ASC_Flow_DP', 'HPC_Suction_Press', 'HPC_Discharge_Press']

    # Convert feature columns to numeric, handling errors
    data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors='coerce')

    # Clean the data
    data_cleaned = data.replace('Bad', np.nan).dropna()

    # Ensure Timestamp is in datetime format
    data_cleaned['Timestamp'] = pd.to_datetime(data_cleaned['Timestamp'], utc=True)

    # Extract time-related features
    data_cleaned['Minute'] = data_cleaned['Timestamp'].dt.minute
    data_cleaned['Hour'] = data_cleaned['Timestamp'].dt.hour
    data_cleaned['Day'] = data_cleaned['Timestamp'].dt.day
    data_cleaned['Month'] = data_cleaned['Timestamp'].dt.month
    data_cleaned['Weekday'] = data_cleaned['Timestamp'].dt.weekday

    # Cyclical encoding for 'Hour' and 'Minute'
    data_cleaned['Hour_sin'] = np.sin(2 * np.pi * data_cleaned['Hour'] / 24)
    data_cleaned['Hour_cos'] = np.cos(2 * np.pi * data_cleaned['Hour'] / 24)
    data_cleaned['Minute_sin'] = np.sin(2 * np.pi * data_cleaned['Minute'] / 60)
    data_cleaned['Minute_cos'] = np.cos(2 * np.pi * data_cleaned['Minute'] / 60)

    # Group by (Day, Hour, Minute) and aggregate
    data_aggregated = data_cleaned.groupby(['Day', 'Hour', 'Minute']).agg(['mean', 'std'])

    # Rename the columns
    data_aggregated.columns = [f'{agg_type.upper()}_{col}' for col, agg_type in data_aggregated.columns]

    # Reset index
    data_aggregated = data_aggregated.reset_index()

    # Create a Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Imputation step
        ('scaler', MinMaxScaler()),  # Scaling step using MinMaxScaler
        ('ocsvm', OneClassSVM(kernel='rbf', gamma='scale', nu=0.02))  # OCSVM step
    ])

    # Prepare features
    features = data_aggregated.drop(['MEAN_Timestamp', 'STD_Timestamp'], axis=1, errors='ignore')

    # Fit the Pipeline
    pipeline.fit(features)

    # Predict anomalies
    data_aggregated['anomaly'] = pipeline.predict(features)

    # Get anomaly scores
    data_aggregated['anomaly_score'] = pipeline.decision_function(features)

    # Convert -1 (anomaly) to 1 and 1 (normal) to -1
    data_aggregated['anomaly'] = data_aggregated['anomaly'].map({1: 1, -1: -1})

    return data_aggregated