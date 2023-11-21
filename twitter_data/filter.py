import pandas as pd

# Read first 100,000 rows
df = pd.read_csv('tweets.csv'   )
print(df.head)