import pandas as pd

def dict_to_csv(data, file_name):
    # Create DataFrame from input dictionary
    df = pd.DataFrame.from_dict(data, orient='index')

    # Try to read existing CSV file
    try:
        old_df = pd.read_csv(file_name)
        df = pd.concat([old_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    # Write DataFrame to CSV file
    df.to_csv(file_name, index=False)

data = {1: {'name': 'John', 'age': 30}, 2: {'name': 'Jane', 'age': 25}}
file_name = 'data.csv'
dict_to_csv(data, file_name)
