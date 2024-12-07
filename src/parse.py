import pandas as pd

csv = "/Users/mohammed/Downloads/dsfinal/healthcare_dataset.csv"

df = pd.read_csv(csv)

print(df.head(20))