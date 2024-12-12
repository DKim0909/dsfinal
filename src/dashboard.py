import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib


class HealthcareDashboard:
    def __init__(self, db_path, model_path):
        try:
            self.conn = sqlite3.connect(db_path)
            self.model_path = model_path

            # Fetch the data
            self.df = pd.read_sql_query("""
                SELECT 
                    p.patient_id,
                    p.age,
                    p.gender,
                    p.blood_type,
                    p.medical_condition,
                    a.admission_date,
                    a.doctor,
                    a.hospital,
                    a.insurance_provider,
                    b.billing_amount,
                    a.room_number,
                    a.admission_type,
                    a.discharge_date,
                    m.medication,
                    t.test_results
                FROM Patients p
                LEFT JOIN Admissions a ON p.patient_id = a.patient_id
                LEFT JOIN Billing b ON a.admission_id = b.admission_id
                LEFT JOIN Medications m ON a.admission_id = m.admission_id
                LEFT JOIN Tests t ON a.admission_id = t.admission_id
                WHERE a.discharge_date IS NOT NULL 
                    AND a.admission_date IS NOT NULL
            """, self.conn)

            print(f"Data loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
            print(f"Total unique patients: {self.df['patient_id'].nunique()}")

            # Load the trained model
            self.encode_categorical_data()
            self.train_model()

        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            raise

    def load_model(self):
        try:
            return load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def gender_cost_correlation(self):
        plt.figure(figsize=(10, 6))
        
        # Calculate average costs by gender
        gender_stats = self.df.groupby('gender')['total_charges'].mean().round(2)
        
        # Create bar plot
        bars = plt.bar(range(len(gender_stats)), gender_stats)
        
        # Add cost annotations
        for i, cost in enumerate(gender_stats):
            plt.text(i, cost, f'${cost:,.2f}',
                    ha='center', va='bottom')
        
        plt.xticks(range(len(gender_stats)), gender_stats.index)
        plt.title('Average Medical Costs by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Average Cost ($)')
        
        return plt

    def condition_stay_correlation(self):
        plt.figure(figsize=(12, 6))
        
        # Calculate average stay by condition
        condition_stats = self.df.groupby('medical_condition').agg({
            'length_of_stay': 'mean',
            'patient_id': 'nunique'
        }).sort_values('length_of_stay')
        
        # Create horizontal bar plot
        y_pos = np.arange(len(condition_stats))
        bars = plt.barh(y_pos, condition_stats['length_of_stay'])
        
        # Set labels
        plt.yticks(y_pos, condition_stats.index)
        
        # Add count annotations
        for i, row in enumerate(condition_stats.itertuples()):
            plt.text(row.length_of_stay + 0.1, i,
                    f'{row.patient_id} patients',
                    va='center')
        
        plt.title('Average Length of Stay by Medical Condition')
        plt.xlabel('Average Length of Stay (Days)')
        
        return plt

    def top_doctors_by_patients(self):
        plt.figure(figsize=(12, 6))
        
        # Group by doctor and count the number of patients
        doctor_patient_counts = self.df.groupby('doctor')['patient_id'].nunique().sort_values(ascending=False)
        
        # Select the top 10 doctors
        top_doctors = doctor_patient_counts.head(10)
        
        # Create a horizontal bar chart
        y_pos = np.arange(len(top_doctors))
        plt.barh(y_pos, top_doctors.values)
        
        # Set labels
        plt.yticks(y_pos, top_doctors.index)
        plt.xlabel('Number of Patients')
        plt.ylabel('Doctor')
        plt.title('Top 10 Doctors by Number of Patients')
        
        return plt

    def admission_type_analysis(self):
        plt.figure(figsize=(12, 6))
        
        # Calculate distribution of admission types by condition
        admission_dist = pd.crosstab(
            self.df['medical_condition'],
            self.df['admission_type'],
            normalize='index'
        ) * 100
        
        # Create stacked bar chart
        admission_dist.plot(kind='barh', stacked=True)
        
        plt.title('Distribution of Admission Types by Medical Condition')
        plt.xlabel('Percentage of Patients')
        plt.ylabel('Medical Condition')
        plt.legend(title='Admission Type', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        return plt


    def predict_future_cost(self, patient_data):
    
        prepared_data = self.prepare_data(patient_data)

    
        predicted_cost = self.model.predict(prepared_data)[0]
        return predicted_cost

    def prepare_data(self, patient_data):
    
        feature_columns = ['age', 'length_of_stay', 'Male', 'Female', 'Cancer', 'Obesity', 'Diabetes', 'Asthma', 'Hypertension', 'Arthritis', 'Urgent', 'Emergency', 'Elective']
        df = pd.DataFrame([patient_data])

    
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  # Set default value if column is missing

        X = df[feature_columns].values
        return X

    
    def encode_categorical_data(self):
    
        encoder = OneHotEncoder()

    
        encoded_gender = encoder.fit_transform(self.df[['gender']]).toarray()
        encoded_medical_condition = encoder.fit_transform(self.df[['medical_condition']]).toarray()
        encoded_admission_type = encoder.fit_transform(self.df[['admission_type']]).toarray()

    
        encoded_gender_df = pd.DataFrame(encoded_gender, columns=encoder.categories_[0])
        encoded_medical_condition_df = pd.DataFrame(encoded_medical_condition, columns=encoder.categories_[1])
        encoded_admission_type_df = pd.DataFrame(encoded_admission_type, columns=encoder.categories_[2])

    
        self.df = pd.concat([self.df, encoded_gender_df, encoded_medical_condition_df, encoded_admission_type_df], axis=1)

    
        self.df.drop(['gender', 'medical_condition', 'admission_type'], axis=1, inplace=True)


    
    def train_model(self):

        features = ['age', 'length_of_stay', 'Male', 'Female', 'Cancer', 'Obesity', 'Diabetes', 'Asthma', 'Hypertension', 'Arthritis', 'Urgent', 'Emergency', 'Elective']
        target = 'total_charges'


        X = self.df[features]
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        self.model = LinearRegression()
        self.model.fit(X_train, y_train)


        y_pred = self.model.predict(X_test)



        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error: {rmse:.2f}")

        return self.model


def main():
    st.set_page_config(layout="wide")
    st.title('Healthcare Analytics Dashboard')

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, '..', 'data', 'healthcare.db')
        model_path = os.path.join(current_dir, '..', 'models', 'linear_regression_model.pkl')

        dashboard = HealthcareDashboard(db_path, model_path)

        # ... (Display key metrics and visualizations as before)

        # Display patient prediction
        patient_id = st.text_input("Enter Patient ID")
        if patient_id:
            patient_data = dashboard.get_patient_data(patient_id)
            predicted_cost = dashboard.predict_cost(patient_data)
            st.write(f"Predicted Future Cost: ${predicted_cost:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"Detailed error: {str(e)}")

if __name__ == "__main__":
    main()