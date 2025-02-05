import numpy as np
import string
import os
import matplotlib
import matplotlib.pyplot as plt

# Visualising and save data.
def visualise_data(data, filename: str):
    plt.scatter(init_data[:, 0], init_data[:, 1])
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}.png")

# Saving the data as CSV.
def save_data_as_csv(data, filename:str):
    os.makedirs("dataFiles", exist_ok=True)
    np.savetxt(f"dataFiles/{filename}.csv", init_data, delimiter=',')

# Augmenting the data.
def augment_data(data, num_samples):
    augmented_data = []
    for _ in range(num_samples):
        # Randomly select a point from the original data.
        point = data[np.random.randint(0, len(data))]
        # Add Gaussian noise to the selected point.
        noise = np.random.normal(0, 0.1, size=point.shape)
        augmented_point = point + noise
        augmented_data.append(augmented_point)
    return np.array(augmented_data)

# Main Execution
if __name__ == "__main__":

    init_data = np.loadtxt('dataFiles/init_data.csv', delimiter=',')
    visualise_data(init_data, "init_data")
    # Generate augmented data.
    augmented_data = augment_data(init_data, 100)
    visualise_data(augmented_data, "augmented_data")
    save_data_as_csv(augmented_data,"augmented_data")


    # complete_init_data = np.loadtxt('dataFiles/comp_init_data.csv', delimiter=',')
    # ig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 12),
    #                          constrained_layout=True)
    # ax = axes[0, 1]
    # ax.set_title("Pretrained Model on Train Data")




# To do:
# 1. Need to plot based on the label, not just x,y scatter plot.
# 2.visualise, save n stuff init data and augment data with decision surface etc.
# 2. complete with plots and stuff the first question.
# 3. complete with plots, and tuning and stuff the second question.


# 4. understand the starter code given.
# 5. 3 and 4the question.