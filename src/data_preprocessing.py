import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def identify_missing_data(data):
    
    missing_summary = data.isnull().sum()
    missing_percentage = (missing_summary / len(data)) * 100
    missing_report = pd.DataFrame({
        'Missing Count': missing_summary,
        'Missing Percentage': missing_percentage
    })
    
    print("Missing Data Report:")
    print(missing_report)

if __name__ == "__main__":
    
    raw_data_path = "/Users/mohammed/Downloads/dsfinal/data/raw.csv"
    raw_data = load_data(raw_data_path)
    
    missing_report = identify_missing_data(raw_data)