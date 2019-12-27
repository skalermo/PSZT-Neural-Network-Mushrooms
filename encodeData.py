import pandas as pd

# Read file
data = pd.read_csv('agaricus-lepiota.csv')

# Print Head
print(data.head(5))

# Check for missing values
for col in data.columns:
    print('{} : {}'.format(col, data[col].isnull().sum()))

# One hot encode data
encoded = pd.get_dummies(data)

print('Number of Columns: {} -> {}'.format(len(data.columns), len(encoded.columns)))

# Save to csv file
encoded.to_csv('temp.csv', index=False)