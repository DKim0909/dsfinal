import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import os

class HealthcareDashboard:
    def __init__(self, db_path):
        try:
            self.conn = sqlite3.connect(db_path)
            
            # Fetch the data
            self.df = pd.read_sql_query("""
                SELECT 
                    p.patient_id,
                    p.gender,
                    p.medical_condition,
                    a.admission_date,
                    a.discharge_date,
                    a.doctor,
                    a.admission_type,
                    COALESCE(CAST(b.total_charges AS FLOAT), 0) as total_charges,
                    CAST((JULIANDAY(a.discharge_date) - JULIANDAY(a.admission_date)) AS INTEGER) as length_of_stay
                FROM Patients p
                LEFT JOIN Admissions a ON p.patient_id = a.patient_id
                LEFT JOIN Billing b ON a.admission_id = b.admission_id
                WHERE a.discharge_date IS NOT NULL 
                    AND a.admission_date IS NOT NULL
            """, self.conn)
            
            print(f"Data loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
            print(f"Total unique patients: {self.df['patient_id'].nunique()}")
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            raise

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

def main():
    st.set_page_config(layout="wide")
    st.title('Healthcare Analytics Dashboard')
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, '..', 'data', 'healthcare.db')
        
        dashboard = HealthcareDashboard(db_path)
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", 
                     f"{dashboard.df['patient_id'].nunique():,}")
        with col2:
            st.metric("Average Length of Stay", 
                     f"{dashboard.df['length_of_stay'].mean():.1f} days")
        with col3:
            st.metric("Average Cost", 
                     f"${dashboard.df['total_charges'].mean():,.2f}")
        
        # Display visualizations
        st.subheader('Cost by Gender Analysis')
        st.pyplot(dashboard.gender_cost_correlation())
        
        st.subheader('Medical Condition and Length of Stay Analysis')
        st.pyplot(dashboard.condition_stay_correlation())
        
        st.subheader('Top 10 Doctors By Patients')
        st.pyplot(dashboard.top_doctors_by_patients())
        
        st.subheader('Admission Types by Medical Condition')
        st.pyplot(dashboard.admission_type_analysis())
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"Detailed error: {str(e)}")

if __name__ == "__main__":
    main()
