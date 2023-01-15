import pickle

# File name
file_name = "data.pkl"

# Open the file for reading
with open(file_name, "rb") as file:
    # Load the data from the file
    data = pickle.load(file)

print("Data loaded from file:", data)
