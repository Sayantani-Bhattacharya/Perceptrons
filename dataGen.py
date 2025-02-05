import pandas as pd
# Create the dataset based on the table in the screenshot
data_table = {
    "x1": [1, 2, 2, 4, 10],
    "x2": [2, 1, 3, 3, 3],
    "y": [1, -1, -1, 1, 1],
    "g(X)": [-0.7, 1.5, -1.1, 0.7, 6.1],
    "h(X)": [-1, 1, -1, 1, 1],
}

# Convert to a DataFrame and save to a CSV file
data_table_path = "classifier_table.csv"
pd.DataFrame(data_table).to_csv(data_table_path, index=False)

data_table_path
