import tensorflow as tf
import numpy as np
import sys

if (len(sys.argv) != 2):
	print("This program takes a single argument as a parameter which corresponds to the temperature in celsius to convert to fahrenheit.")
	sys.exit()

value = int(sys.argv[1])

# Array of features
celsius = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype = float)
# Array of labels
fahrenheit = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype = float)

# Print examples 
for i, x in enumerate(celsius):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(x, fahrenheit[i]))

# Create a layer of only one neuron
layer_0 = tf.keras.layers.Dense(units = 1, input_shape = [1])
layers = [ layer_0 ]

# Create a neural network based on our layers
model = tf.keras.Sequential(layers)

# Compile the neural network using default loss and optimization functions
model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))

# Try to predict without any training
print(model.predict([value]))

# Train the network 50 times
history = model.fit(celsius, fahrenheit, epochs = 50, verbose = False)

# The prediction should be better
print(model.predict([value]))

# Train the network 1000 times
history = model.fit(celsius, fahrenheit, epochs = 1000, verbose = False)

# The prediction must be good
print(model.predict([value]))