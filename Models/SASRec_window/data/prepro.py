import pandas as pd

# Load the data
df = pd.read_csv('ratings.csv')

# Sort by 'userId' and 'timestamp'
df_sorted = df.sort_values(by=['userId', 'timestamp'])

# Save the sorted dataframe
df_sorted.to_csv('ml-20m.txt', sep='\t', index=False)