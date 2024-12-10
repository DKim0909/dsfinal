import pandas as pd
import os
import sqlite3
from scipy.stats import zscore

# Step 1: Data Cleaning Functions
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
    print("Duplicate rows found:")
    print(duplicate_rows)
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
    print(f"Unique values in the '{column_name}' column:")
    print(unique_values)
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
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

# Step 2: Database Setup Functions
def create_database(database_path):
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Create Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Patients (
                patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                gender TEXT,
                blood_type TEXT,
                medical_condition TEXT
            );
        """)

        # Create Admissions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Admissions (
                admission_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                admission_date DATE NOT NULL,
                discharge_date DATE,
                admission_type TEXT,
                room_number INTEGER,
                doctor TEXT,
                hospital TEXT,
                FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
            );
        """)

        # Create Billing table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Billing (
                billing_id INTEGER PRIMARY KEY AUTOINCREMENT,
                admission_id INTEGER NOT NULL,
                total_charges DECIMAL(10, 2),
                insurance_provider TEXT,
                FOREIGN KEY (admission_id) REFERENCES Admissions(admission_id)
            );
        """)

        conn.commit()
        print("Database and tables created successfully.")
    except sqlite3.Error as e:
        print(f"Error creating database: {e}")
    finally:
        conn.close()

def populate_database(database_path, cleaned_csv_path):
    try:
        conn = sqlite3.connect(database_path)
        df = pd.read_csv(cleaned_csv_path)

        # Rename columns to match database schema
        df.rename(columns={
            'Blood Type': 'blood_type',
            'Medical Condition': 'medical_condition',
            'Date of Admission': 'admission_date',
            'Discharge Date': 'discharge_date',
            'Admission Type': 'admission_type',
            'Room Number': 'room_number',
            'Doctor': 'doctor',
            'Hospital': 'hospital',
            'Billing Amount': 'total_charges',
            'Insurance Provider': 'insurance_provider'
        }, inplace=True)

        # Populate Patients table
        patients_data = df[['Name', 'Gender', 'blood_type', 'medical_condition']].drop_duplicates()
        patients_data.to_sql('Patients', conn, if_exists='append', index=False)

        # Fetch patient IDs for Admissions table
        cursor = conn.cursor()
        cursor.execute("SELECT patient_id, name FROM Patients;")
        patient_mapping = {row[1]: row[0] for row in cursor.fetchall()}

        # Populate Admissions table
        admissions_data = df[['Name', 'admission_date', 'discharge_date', 'admission_type', 'room_number', 'doctor', 'hospital']].drop_duplicates()
        admissions_data['patient_id'] = admissions_data['Name'].map(patient_mapping)
        admissions_data.drop(columns=['Name'], inplace=True)
        admissions_data.to_sql('Admissions', conn, if_exists='append', index=False)

        # Fetch admission IDs for Billing table
        cursor.execute("SELECT admission_id, admission_date FROM Admissions;")
        admission_mapping = {row[1]: row[0] for row in cursor.fetchall()}

        # Populate Billing table
        billing_data = df[['admission_date', 'total_charges', 'insurance_provider']].drop_duplicates()
        billing_data['admission_id'] = billing_data['admission_date'].map(admission_mapping)
        billing_data.drop(columns=['admission_date'], inplace=True)
        billing_data.to_sql('Billing', conn, if_exists='append', index=False)

        conn.commit()
        print("Database populated successfully.")
    except sqlite3.Error as e:
        print(f"Error populating database: {e}")
    finally:
        conn.close()

# Main script
if __name__ == "__main__":
    # Set paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(project_root, "../data/raw.csv")
    cleaned_data_path = os.path.join(project_root, "../data/cleaned.csv")
    database_path = os.path.join(project_root, "../data/healthcare.db")

    # Step 1: Data Cleaning
    raw_data = load_data(raw_data_path)
    identify_missing_data(raw_data)
    find_duplicate_rows(raw_data)
    raw_data, _ = drop_duplicates(raw_data)
    raw_data = format_name(raw_data)
    raw_data = format_date(raw_data)
    for column in ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication', 'Test Results']:
        find_unique_values(raw_data, column)
    raw_data = remove_outliers(raw_data, 'Billing Amount')
    find_admission_dates(raw_data, 'Date of Admission')
    find_admission_dates(raw_data, 'Discharge Date')
    save_data(raw_data, cleaned_data_path)
    print(f"Cleaned data saved to {cleaned_data_path}")

    # Step 2: Database Setup
    create_database(database_path)

    # Step 3: Populate Database
    populate_database(database_path, cleaned_data_path)

    print("All steps completed successfully.")
