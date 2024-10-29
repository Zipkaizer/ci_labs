from minisom import MiniSom
import numpy as np
import random


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

# Define the range of learning rates to test
learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]

# Initialize the SOM with a 2x2 grid and adjusted parameters
som = MiniSom(x=2, y=2, input_len=11, sigma=0.3, learning_rate=0.5, random_seed=50)
som.random_weights_init(data)

# the number of training iterations
som.train_random(data, num_iteration=200)

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
for i, pos in enumerate(cluster_assignments, start=1):
    cluster = cluster_labels[pos]
    print(f"Vector v{i} is assigned to cluster {cluster} at position {pos}")


# Initialize a dictionary to store results
results = {}

for lr in learning_rates:
    print(f"\nTesting with learning rate: {lr}")
    # Initialize the SOM with the current learning rate
    som = MiniSom(x=2, y=2, input_len=11, sigma=0.5, learning_rate=lr, random_seed=50)
    som.random_weights_init(data)
    # Train the SOM
    som.train_random(data, num_iteration=100)
    # Assign vectors to clusters
    cluster_assignments = [som.winner(datum) for datum in data]
    # Convert positions to standard integers
    cluster_assignments = [(int(pos[0]), int(pos[1])) for pos in cluster_assignments]
    # Map positions to cluster labels
    cluster_labels = {}
    label = 0
    for pos in sorted(set(cluster_assignments)):
        cluster_labels[pos] = label
        label += 1
    # Print cluster assignments
    for i, pos in enumerate(cluster_assignments, start=1):
        cluster = cluster_labels[pos]
        print(f"Vector v{i} is assigned to cluster {cluster} at position {pos}")
    # Store the results
    results[lr] = cluster_assignments

