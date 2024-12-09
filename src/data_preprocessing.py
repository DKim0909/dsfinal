import pandas as pd
import os
from scipy import stats
from scipy.stats import zscore


def load_data(file_path):
    return pd.read_csv(file_path)

def identify_missing_data(df):
    missing_summary = df.isnull().sum()
    missing_percentage = (missing_summary / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Count': missing_summary,
        'Missing Percentage': missing_percentage
    })
    print("Missing Data Report:")
    print(missing_report)
    return missing_report

def drop_duplicates(df):
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicates before dropping: {duplicate_count}")
    df = df.drop_duplicates()
    return df, duplicate_count

def find_duplicate_rows(df):
    duplicate_mask = df.duplicated(keep=False)
    duplicate_rows = df[duplicate_mask]
    return duplicate_rows

def format_name(df):
    df['Name'] = df['Name'].str.title()
    return df

def format_date(df):
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
    return df

def find_unique_values(df, column_name):
    unique_values = df[column_name].unique()
    return unique_values

def remove_outliers(df, column_name, threshold=3):
    df['z_score'] = zscore(df[column_name])
    original_row_count = len(df)
    df_cleaned = df[df['z_score'].abs() <= threshold]
    cleaned_row_count = len(df_cleaned)
    rows_affected = original_row_count - cleaned_row_count
    df_cleaned = df_cleaned.drop(columns=['z_score'])
    print(f"Number of rows affected by removing outliers: {rows_affected}")
    return df_cleaned

def find_admission_dates(df, column_name):
    earliest_date = df[column_name].min()
    most_recent_date = df[column_name].max()
    print(f"Earliest date: {earliest_date}")
    print(f"Most recent date: {most_recent_date}")

def save_data(df, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    # Set relative paths based on the project structure
    project_root = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    raw_data_path = os.path.join(project_root, "../data/raw.csv")
    cleaned_data_path = os.path.join(project_root, "../data/cleaned.csv")

    # Load raw data
    raw_data = load_data(raw_data_path)

    # Identify missing data
    missing_report = identify_missing_data(raw_data)

    # Find duplicate rows
    duplicate_rows = find_duplicate_rows(raw_data)
    print("Duplicate rows found:")
    print(duplicate_rows)

    # Drop duplicates
    raw_data, duplicate_count = drop_duplicates(raw_data)
    print(f"Number of duplicates removed: {duplicate_count}")

    # Format columns
    raw_data = format_name(raw_data)
    raw_data = format_date(raw_data)

    # Find unique values in categorical columns
    for column in ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication', 'Test Results']:
        unique_values = find_unique_values(raw_data, column)
        print(f"Unique values in the '{column}' column:")
        print(unique_values)

    # Remove outliers
    raw_data = remove_outliers(raw_data, 'Billing Amount')

    # Find admission dates
    find_admission_dates(raw_data, 'Date of Admission')
    find_admission_dates(raw_data, 'Discharge Date')

    # Save cleaned data
    save_data(raw_data, cleaned_data_path)
    print(f"Cleaned data saved to {cleaned_data_path}")
