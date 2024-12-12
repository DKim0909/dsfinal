# Patient Outcome Prediction Project - README

## Project Overview

Our project focuses on predicting patient outcomes, such as recovery times and readmission risks, using healthcare data. We aim to apply data science techniques to improve patient care and optimize hospital resource management. The project involves several key steps: cleaning and organizing healthcare data, designing an efficient SQL database for data storage, and employing regression models to identify important patterns. By applying course concepts such as data preprocessing, regression, and database design, we are translating theoretical knowledge into practical use in a real-world healthcare setting.

## How to Run the Dashboard

First, run `data_preprocessing.py` to clean data and load into`healthcare.db` In order to run the dashboard, follow these steps:

1. Open a terminal and navigate to the `/dsfinal/src` directory.
   
   ```
   cd /dsfinal/src
   ```

2. Run the Install for the required libraries and then Streamlit application using the following commands:

   ```
   pip install pandas numpy matplotlib seaborn streamlit scikit-learn
   streamlit run dashboard.py
   ```

3. The dashboard will open in your default web browser, where you can interact with the data and explore model predictions.
