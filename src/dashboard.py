import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import os
from sklearn.linear_model import LinearRegression

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

    def predict_future_patients(self):
        """Predict future patient numbers using time series analysis with multiple growth factors"""
        from sklearn.linear_model import LinearRegression
        import datetime
        
        # Convert admission dates to datetime if not already
        self.df['admission_date'] = pd.to_datetime(self.df['admission_date'])
        
        # Group by month and count patients
        monthly_patients = self.df.groupby(pd.Grouper(key='admission_date', freq='M'))['patient_id'].nunique()
        monthly_patients = monthly_patients[:-1]  # Remove last incomplete month
        
        # Base prediction using linear regression
        X = np.arange(len(monthly_patients)).reshape(-1, 1)
        y = monthly_patients.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next 120 months (10 years)
        future_months = np.arange(len(monthly_patients), len(monthly_patients) + 120).reshape(-1, 1)
        base_prediction = model.predict(future_months)
        
        # Growth factors (annual rates)
        population_growth_rate = 0.007  # US population growth ~0.7% per year
        obesity_growth_rate = 0.015     # Obesity rate growth ~1.5% per year
        diabetes_growth_rate = 0.013    # Diabetes growth ~1.3% per year
        aging_population_rate = 0.01    # Population aging effect ~1% per year
        mortality_rate = -0.005         # General mortality rate -0.5% per year
        
        # Combined monthly growth rate
        total_annual_growth = (population_growth_rate + obesity_growth_rate + 
                             diabetes_growth_rate + aging_population_rate + mortality_rate)
        monthly_growth_rate = total_annual_growth / 12
        
        # Apply compound growth
        months = np.arange(120)
        growth_factors = (1 + monthly_growth_rate) ** months
        final_prediction = base_prediction * growth_factors
        
        # Create visualization
        plt.figure(figsize=(15, 7))
        
        # Plot historical data
        plt.plot(monthly_patients.index, monthly_patients.values, 
                label='Historical Data', color='blue')
        
        # Plot prediction
        future_dates = pd.date_range(start=monthly_patients.index[-1], 
                                    periods=121, freq='M')[1:]
        plt.plot(future_dates, final_prediction, 
                label='Predicted', color='red', linestyle='--')
        
        # Add confidence interval
        std_dev = np.std(y)
        confidence_interval = std_dev * np.sqrt(1 + np.arange(len(final_prediction))/len(y))
        plt.fill_between(future_dates,
                        final_prediction - confidence_interval,
                        final_prediction + confidence_interval,
                        color='red', alpha=0.2,
                        label='Prediction Interval')
        
        plt.title('10-Year Patient Volume Prediction with Growth Trends')
        plt.xlabel('Date')
        plt.ylabel('Number of Patients per Month')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate and format growth metrics
        current_avg = monthly_patients.mean()
        predicted_avg_1yr = final_prediction[:12].mean()
        predicted_avg_5yr = final_prediction[48:60].mean()
        predicted_avg_10yr = final_prediction[-12:].mean()
        
        growth_1yr = ((predicted_avg_1yr - current_avg) / current_avg) * 100
        growth_5yr = ((predicted_avg_5yr - current_avg) / current_avg) * 100
        growth_10yr = ((predicted_avg_10yr - current_avg) / current_avg) * 100
        
        # Add detailed annotations
        annotation_text = (
            f'Current monthly average: {current_avg:.0f}\n'
            f'1-year prediction: {predicted_avg_1yr:.0f} ({growth_1yr:+.1f}%)\n'
            f'5-year prediction: {predicted_avg_5yr:.0f} ({growth_5yr:+.1f}%)\n'
            f'10-year prediction: {predicted_avg_10yr:.0f} ({growth_10yr:+.1f}%)\n\n'
            f'Growth Factors Included:\n'
            f'• Population Growth: {population_growth_rate*100:.1f}%/yr\n'
            f'• Obesity Trend: {obesity_growth_rate*100:.1f}%/yr\n'
            f'• Diabetes Trend: {diabetes_growth_rate*100:.1f}%/yr\n'
            f'• Aging Effect: {aging_population_rate*100:.1f}%/yr\n'
            f'• Mortality Rate: {mortality_rate*100:.1f}%/yr'
        )
        
        plt.text(0.02, 0.98, annotation_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        return plt, model, {
            'current_avg': current_avg,
            'predicted_1yr': predicted_avg_1yr,
            'predicted_5yr': predicted_avg_5yr,
            'predicted_10yr': predicted_avg_10yr,
            'growth_1yr': growth_1yr,
            'growth_5yr': growth_5yr,
            'growth_10yr': growth_10yr
        }

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
        
        # Prediction Section
        st.subheader('Future Patient Volume Prediction')
        fig, model, predictions = dashboard.predict_future_patients()
        st.pyplot(fig)
        
        # Display prediction metrics with %s
        met_col1, met_col2, met_col3 = st.columns(3)
        with met_col1:
            st.metric("Current Monthly Average", 
                     f"{predictions['current_avg']:.0f} patients")
        with met_col2:
            st.metric("1-Year Prediction", 
                     f"{predictions['predicted_1yr']:.0f} patients",
                     f"{predictions['growth_1yr']:+.1f}%")
        with met_col3:
            st.metric("5-Year Prediction",
                     f"{predictions['predicted_5yr']:.0f} patients",
                     f"{predictions['growth_5yr']:+.1f}%")

        met_col4, met_col5, met_col6 = st.columns(3)
        with met_col5:
            st.metric("10-Year Prediction",
                     f"{predictions['predicted_10yr']:.0f} patients",
                     f"{predictions['growth_10yr']:+.1f}%")

        # Explanation
        st.write("""
        ### Prediction Model Factors
        This 10-year forecast incorporates multiple growth factors:
        - US Population Growth Rate
        - Disease Prevalence Trends (Obesity, Diabetes)
        - Aging Population Effects
        - Mortality Rates
        - Historical Patient Data Patterns

        The model factors in each rate above and predits the growth rate over the next 10 years. The longer the term the higher uncertainty. 
        """)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"Detailed error: {str(e)}")

if __name__ == "__main__":
    main()