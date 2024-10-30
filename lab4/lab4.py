# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Step 1 & 2: Prepare the data
# Create input and output datasets
inputs = []
outputs = []

for X1 in [0, 1]:
    for X2 in [0, 1]:
        for X3 in [0, 1]:
            for X4 in [0, 1]:
                input_vector = [X1, X2, X3, X4]
                Y = (((X1 or X2) and X3) or X4)
                inputs.append(input_vector)
                outputs.append([int(Y)])

inputs = np.array(inputs)
outputs = np.array(outputs)

# Step 3: Create the neural network
# Define the neural network model
model = Sequential()

# Hidden layer with 2 neurons and 'tanh' activation function
model.add(Dense(units=2, activation='tanh', input_dim=4,
                kernel_initializer='glorot_uniform', bias_initializer='zeros'))

# Output layer with 1 neuron and 'sigmoid' activation function
model.add(Dense(units=1, activation='sigmoid',
                kernel_initializer='glorot_uniform', bias_initializer='zeros'))

# Compile the model
optimizer = SGD(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Step 4: Train the neural network
epochs = 2000
history = model.fit(inputs, outputs, epochs=epochs, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(inputs, outputs, verbose=0)
print(f"\nTraining Loss: {loss:.4f}")
print(f"Training Accuracy: {accuracy:.4f}")

# Access the weights and biases
for layer_num, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    print(f"\nLayer {layer_num + 1} weights:")
    print(weights)
    print(f"Layer {layer_num + 1} biases:")
    print(biases)

# Plot training loss over epochs
plt.plot(history.history['loss'])
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot training accuracy over epochs
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Step 5: Test the neural network with specific inputs
def test_neural_network(model, X1, X2, X3, X4):
    # Prepare the input vector
    input_vector = np.array([[X1, X2, X3, X4]])

    # Compute the expected output using the logical expression
    expected_output = int((((X1 or X2) and X3) or X4))

    # Get the neural network's prediction
    prediction = model.predict(input_vector, verbose=0)
    predicted_output = int((prediction > 0.5)[0][0])

    # Print the results
    print(f"\nTesting Neural Network with inputs X1={X1}, X2={X2}, X3={X3}, X4={X4}")
    print(f"Expected Output: {expected_output}")
    print(f"Neural Network Predicted Output: {predicted_output}")

    # Check if the prediction is correct
    if predicted_output == expected_output:
        print("The neural network's prediction is correct.")
    else:
        print("The neural network's prediction is incorrect.")

# Example tests
test_neural_network(model, 0, 0, 0, 0)
test_neural_network(model, 1, 1, 1, 0)
test_neural_network(model, 0, 1, 0, 1)
test_neural_network(model, 1, 0, 1, 0)
