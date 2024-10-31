from minisom import MiniSom
import numpy as np
import random
import time  # For measuring training time

# Define vowels and names as before
vowels = set('аеиіоуияєюєїАЕИІОУИЯЄЮЄЇ')

surname = 'Максименков'
first_name = 'Олексій'
patronymic = 'Юрійович'

# Store the names in a dictionary
names = {
    'surname': surname,
    'first_name': first_name,
    'patronymic': patronymic
}

# Find the word with the largest number of letters
max_length = max(len(name) for name in names.values())
# Collect all words that have the maximum length (in case of ties)
max_length_words = [name for name in names.values() if len(name) == max_length]
# Select one of them
longest_word = max_length_words[0]

print(f"The word with the largest number of letters is: {longest_word} (Length: {max_length})")

# Function to encode a word into a binary vector
def encode_word(word, target_length):
    vector = [1 if char in vowels else 0 for char in word]
    # Adjust length by adding zeros to the left if necessary
    if len(vector) < target_length:
        vector = [0] * (target_length - len(vector)) + vector
    return vector

# Use the length of the longest word as the target length
target_length = max_length

v1 = encode_word(longest_word, target_length)

v2 = v1.copy()
v2[1] = 1 - v2[1]  # Invert the second bit

v3 = v1.copy()
v3[2] = 1 - v3[2]  # Invert the third bit

# Get the other two words
other_words = [name for name in names.values() if name != longest_word]

v4 = encode_word(other_words[0], target_length)
v5 = encode_word(other_words[1], target_length)

v6 = v4.copy()
v6[1] = 1 - v6[1]  # Invert the second bit

v7 = v4.copy()
v7[2] = 1 - v7[2]  # Invert the third bit

v8 = v5.copy()
v8[1] = 1 - v8[1]  # Invert the second bit

v9 = v5.copy()
v9[2] = 1 - v9[2]  # Invert the third bit

# Collect all vectors
vectors = [v1, v2, v3, v4, v5, v6, v7, v8, v9]

for i, v in enumerate(vectors, start=1):
    print(f"v{i} (length {len(v)}): {v}")

data = np.array(vectors)

# Set the random seed for reproducibility
np.random.seed(50)
random.seed(50)

# Define decay functions
def learning_rate_decay(initial_lr, i, max_iter):
    return initial_lr * np.exp(-i / max_iter)

def sigma_decay(initial_sigma, i, max_iter):
    return initial_sigma * np.exp(-i / max_iter)

# Implement decaying learning rate and sigma in a custom training function
def train_som(som, data, num_iteration):
    initial_lr = som._learning_rate
    initial_sigma = som._sigma
    for i in range(num_iteration):
        # Update learning rate and sigma
        som._learning_rate = learning_rate_decay(initial_lr, i, num_iteration)
        som._sigma = sigma_decay(initial_sigma, i, num_iteration)
        # Randomly select a data sample
        idx = np.random.randint(0, len(data))
        x = data[idx]
        # Find BMU and update weights
        bmu = som.winner(x)
        som.update(x, bmu, i, num_iteration)

# Initialize the SOM with adjusted parameters
som = MiniSom(x=1, y=3, input_len=11, sigma=1.0, learning_rate=0.5, random_seed=50)
som.random_weights_init(data)

# Measure training time
start_time = time.time()
train_som(som, data, num_iteration=1000)
training_time = time.time() - start_time

print(f"\nTraining time for initial model: {training_time:.4f} seconds")

# Compute Quantization Error
quantization_error = som.quantization_error(data)
print(f"Quantization Error for initial model: {quantization_error:.4f}")

# Compute Topographic Error
def topographic_error(som, data):
    error = 0
    for x in data:
        bmu = som.winner(x)
        # Compute distances to all neurons
        distances = np.linalg.norm(som._weights - x, axis=-1)
        # Set distance to BMU to infinity to find second BMU
        distances[bmu] = np.inf
        bmu_2 = np.unravel_index(np.argmin(distances), distances.shape)
        if not is_adjacent(bmu, bmu_2):
            error += 1
    return error / len(data)

def is_adjacent(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return max(abs(x1 - x2), abs(y1 - y2)) == 1

topo_error = topographic_error(som, data)
print(f"Topographic Error for initial model: {topo_error:.4f}")

# Assign each vector to a cluster
cluster_assignments = [som.winner(datum) for datum in data]

# Convert positions to standard integers for cleaner output
cluster_assignments = [(int(pos[0]), int(pos[1])) for pos in cluster_assignments]

# Map the 2D positions to cluster labels
cluster_labels = {}
label = 0
for pos in sorted(set(cluster_assignments)):
    cluster_labels[pos] = label
    label += 1

# Print cluster assignments
print("\nCluster assignments for initial model:")
for i, pos in enumerate(cluster_assignments, start=1):
    cluster = cluster_labels[pos]
    print(f"Vector v{i} is assigned to cluster {cluster} at position {pos}")

# Testing Different Learning Rates with Adjustments
learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]

results = {}

for lr in learning_rates:
    print(f"\nTesting with learning rate: {lr}")
    som = MiniSom(x=1, y=3, input_len=11, sigma=1.0, learning_rate=lr, random_seed=50)
    som.random_weights_init(data)
    # Measure training time
    start_time = time.time()
    train_som(som, data, num_iteration=1000)
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.4f} seconds")
    quantization_error = som.quantization_error(data)
    print(f"Quantization Error: {quantization_error:.4f}")
    topo_error = topographic_error(som, data)
    print(f"Topographic Error: {topo_error:.4f}")
    cluster_assignments = [som.winner(datum) for datum in data]
    cluster_assignments = [(int(pos[0]), int(pos[1])) for pos in cluster_assignments]
    cluster_labels = {}
    label = 0
    for pos in sorted(set(cluster_assignments)):
        cluster_labels[pos] = label
        label += 1
    for i, pos in enumerate(cluster_assignments, start=1):
        cluster = cluster_labels[pos]
        print(f"Vector v{i} is assigned to cluster {cluster} at position {pos}")
    results[lr] = {
        'training_time': training_time,
        'quantization_error': quantization_error,
        'topographic_error': topo_error,
        'cluster_assignments': cluster_assignments
    }
