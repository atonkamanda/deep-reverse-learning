import pickle

# Sample list
data = [1, 2, 3, 4, 5]

# File name
file_name = "data.pkl"

# Open the file for writing
with open(file_name, "wb") as file:
    # Dump the data to the file
    pickle.dump(data, file)

print("Data written to file:", file_name)
