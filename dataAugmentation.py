import numpy as np
import string
import os
import matplotlib
import matplotlib.pyplot as plt

# Visualising and save data.
def visualise_data(data, filename: str):
    plt.scatter(data[:, 0],data[:, 1], c=data[:,2], cmap="coolwarm", edgecolors="k")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}.png")

# Saving the data as CSV.
def save_data_as_csv(data, filename:str):
    os.makedirs("dataFiles", exist_ok=True)
    np.savetxt(f"dataFiles/{filename}.csv", init_data, delimiter=',')

# To generate an random linearly separable dataset.
def augment_uniformly():
    np.random.seed(42)
    augmented_data = []
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)  # Label: 1 if x1 + x2 > 0, else -1: to make it seperable.
    augmented_data = np.column_stack((X, y))
    return augmented_data

#  To generate a dataset by adding gusian noise to the initial dataset of 5 samples.
def augment_gausian(data, num_samples):
    augmented_data = []
    for _ in range(num_samples):
        # Randomly select a point from the original data.
        point = data[np.random.randint(0, len(data))]
        # Add Gaussian noise to the selected point.
        noise = np.random.normal(0, 1, size=point.shape)
        augmented_point = point + noise
        augmented_data.append(augmented_point)
        # the label also needs to be set.
    return np.array(augmented_data)

# absolutely randomness.
def augment_randomly(data, num_samples):
    augmented_data = []
    max_Xvalue = np.max(data[:, 0])
    min_Xvalue = np.min(data[:, 0])
    max_Yvalue = np.max(data[:, 1])
    min_Yvalue = np.min(data[:, 1])
    X = np.random.uniform(min_Xvalue, max_Xvalue, num_samples)
    Y = np.random.uniform(min_Yvalue, max_Yvalue, num_samples)
    y = np.random.choice([-1, 1], size=num_samples)
    augmented_data = np.column_stack((X,Y,y))
    return augmented_data

# Main Execution
if __name__ == "__main__":

    init_data = np.loadtxt('dataFiles/init_data.csv', delimiter=',')
    visualise_data(init_data, "init_data")

    # Generate augmented data.
    # augmented_data = augment_uniformly()
    # augmented_data = augment_gausian(init_data,100)
    augmented_data = augment_randomly(init_data,100)
    
    visualise_data(augmented_data, "augmented_data")
    save_data_as_csv(augmented_data,"augmented_data")


    


