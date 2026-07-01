import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- SECTION 1: Load Dataset ---
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape}")
    return df

# --- SECTION 2: Data Type Verification ---
def enforce_types(df):
    print("Enforcing data types...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Store'] = df['Store'].astype(int)
    df['Dept'] = df['Dept'].astype(int)
    df['IsHoliday'] = df['IsHoliday'].astype(int)
    df['Type'] = df['Type'].astype('category')
    return df

# --- SECTION 3-12: Feature Engineering Functions ---
def create_time_features(df):
    print("Creating date-based features...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def create_time_series_features(df):
    print("Creating Lag and Rolling features (Grouped by Store/Dept)...")
    # Sort for time-series integrity
    df = df.sort_values(['Store', 'Dept', 'Date'])
    
    # Lag Features
    for lag in [1, 2, 4, 8, 12, 26, 52]:
        df[f'Lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
    
    # Rolling Statistics
    g = df.groupby(['Store', 'Dept'])['Weekly_Sales']
    df['Rolling_Mean_4'] = g.rolling(window=4).mean().reset_index(0, drop=True)
    df['Rolling_Std'] = g.rolling(window=4).std().reset_index(0, drop=True)
    return df

def create_markdown_features(df):
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df['Total_Markdown'] = df[markdown_cols].sum(axis=1)
    df['Number_of_Markdowns'] = df[markdown_cols].gt(0).sum(axis=1)
    return df

def cyclical_encoding(df):
    print("Applying cyclical encoding...")
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    return df

# --- SECTION 16: Outlier Handling ---
def flag_outliers(df, col='Weekly_Sales'):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df['Outlier_Flag'] = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).astype(int)
    return df

# --- SECTION 17: Feature Scaling ---
def scale_features(df, features_to_scale, output_path='scaler.pkl'):
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_data = df.copy()
    scaled_data[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    with open(output_path, 'wb') as f:
        pickle.dump(scaler, f)
    return scaled_data

# --- MAIN EXECUTION PIPELINE ---
def main():
    # Load
    raw_path = r"C:\Users\amans\Downloads\Codtech_Projects\predictive_sales_forecast\merged\merged_sales.csv"
    df = load_data(raw_path)
    df = enforce_types(df)
    
    # Feature Engineering
    df = create_time_features(df)
    df = create_markdown_features(df)
    df = create_time_series_features(df)
    df = cyclical_encoding(df)
    df = flag_outliers(df)
    
    # Categorical Encoding
    df['Type_Label'] = LabelEncoder().fit_transform(df['Type'])
    df = pd.get_dummies(df, columns=['Type'], prefix='Type')
    
    # Missing Values (Last Resort)
    df = df.fillna(method='ffill').fillna(0)
    
    # Save
    processed_path = r"C:\Users\amans\Downloads\Codtech_Projects\predictive_sales_forecast\processed\feature_dataset.csv"
    df.to_csv(processed_path, index=False)
    
    print(f"Feature engineering complete. Saved to {processed_path}")

if __name__ == "__main__":
    main()