import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def read_csv_to_df(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Return the first row 
    return df 

def mean_std_plot(df):
    # Take the first 10 rows of the dataframe
    df_subset = df.head(10)
    columns = df.columns
    print(columns)
    # Remove Entropy and sleep accuracy
    columns = columns.drop(['Entropy', 'Sleep accuracy'])
    # For each row in each column the DataFrame transform the string into a list of floats
    for column in columns:
        df[column] = df[column].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    # Compute the mean of the 'Wake loss' column
    mean = df_subset['Wake loss'].mean()
    
    # Compute the standard deviation of the 'Wake loss' column
    std = df_subset['Wake loss'].std()
    
    # Plot the mean and std using matplotlib
    plt.bar(['Mean', 'Std'], [mean, std])
    plt.show()
    
# Read log.csv into a DataFrame
df = read_csv_to_df('log.csv')
# Call the function
mean_std_plot(df)
