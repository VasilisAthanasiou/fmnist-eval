import tensorflow as tf
import numpy as np
from app.src.classifier import mnist_reader
from matplotlib import pyplot as plt
from time import time

# ------------------------------------------ Dataset operations ---------------------------------------------------------------------------------------------- #
def load_dataset(path='data/fashion'):
    """
    Uses the load_mnist method from mnist_reader.py to split f-mnist data into train/test data and labels
    :param path: path to dataset
    :return: xtrain, ytrain, xtest, ytest
    """
    xtrain, ytrain = mnist_reader.load_mnist(path, kind='train')  # Extract f-mnist data using load_mnist from the mnist repository
    xtest, ytest = mnist_reader.load_mnist(path, kind='t10k')  # https://github.com/zalandoresearch/fashion-mnist
    # Normalize
    xtrain, xtest = xtrain / 255.0, xtest / 255.0
    return xtrain, ytrain, xtest, ytest


def print_images(images):
    for i in range(9):
        plt.subplot(331 + i)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))
    plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# --------------------------------------------------- Create Neural Network ---------------------------------------------------------------------------------- #

def net_init():
    """
    Initializes a simple neural network
    :return: Sequential model
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# --------------------------------------------------- Generate Different Neural Networks --------------------------------------------------------------------- #

def generate_net(hidden_layers=2, n_neurons=128):
    """
    This method is used to generate a Sequential model using given parameters
    :param hidden_layers: Number of hidden layers
    :param n_neurons: Number of neurons per layer
    :return: Sequential model
    """
    # Initialize Sequential model
    model = tf.keras.models.Sequential()

    # Add input layer
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Add hidden layers
    for i in range(hidden_layers):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    # Add output layer
    model.add(tf.keras.layers.Dense(10))

    model.summary()

    return model
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------- Evaluate Multiple Different Neural Networks -------------------------------------------------------------------- #
def evaluate_nets(x_train, y_train, x_test, y_test, eval_layers=False, eval_neurons=False, eval_epochs=False, eval_training_size=False, n_nets=10):
    """
    This method creates multiple different neural networks and evaluates them based on classification accuracy, training time and inference time.
    The results of the evaluation are stored in app/data/experiment-results.txt file
    :param x_train: Train data
    :param y_train: Train labels
    :param x_test: Test data
    :param y_test: Test labels
    :param eval_layers: Determines whether to try different hidden layer configurations
    :param eval_neurons: Determines whether to try different neuron per layer configurations
    :param eval_epochs: Determines whether to try different number of gradient-descent epochs configurations
    :param eval_training_size: Determines whether to try different training data size configurations
    :param n_nets: Number of different NNs examined each time an evaluation is performed
    :return:
    """
    file = open('data/experiment-results.txt', 'a')
    # Set training parameters
    h_layers = 2
    n_neurons = 128
    epochs = 10


    # Examine different hidden layer configurations
    if eval_layers:
        file.write('--------------------------------- Layer Evaluation - | 128 neurons per layer | 10 epochs | 60000 training samples -----------------------\n')
        for h_layers in range(n_nets):
            model = generate_net(hidden_layers=h_layers + 1)  # Initialize network
            acc, train_time, average_inf_time = train_and_infer(model, x_train, y_train, x_test, y_test, epochs)  # Train and infer network
            file.write('For {} hidden layers, the network achieved {:.2f}% classification accuracy, with {:.2f}s training time and {:.2f}s average inference time\n'.format(h_layers + 1, acc * 100, train_time, average_inf_time))
        file.write('-----------------------------------------------------------------------------------------------------------------------------\n')
        model = None
        h_layers = 2
    # Examine different neurons per layer configurations
    if eval_neurons:
        file.write('-------------------------------- Neuron Evaluation - | 2 hidden layers | 10 epochs | 60000 training samples ----------------------------\n')
        for neuron_multiplier in range(n_nets):
            n_neurons = 2 ** (neuron_multiplier + 1)  # Each iteration the number of neurons is a power of 2
            model = generate_net(n_neurons=n_neurons)  # Initialize network
            acc, train_time, average_inf_time = train_and_infer(model, x_train, y_train, x_test, y_test, epochs)  # Train and infer network
            file.write('For {} neurons per layer, the network achieved {:.2f}% classification accuracy, with {:.2f}s training time and {:.2f}s average inference time\n'.format(n_neurons, acc * 100, train_time, average_inf_time))
        file.write('-----------------------------------------------------------------------------------------------------------------------------\n')
        model = None
        n_neurons = 128
    if eval_epochs:
        file.write('-------------------------------- Epoch Evaluation - | 2 hidden layers | 128 neurons per layer | 60000 training samples ----------------------------\n')
        for epochs in range(n_nets):
            model = generate_net()
            acc, train_time, average_inf_time = train_and_infer(model, x_train, y_train, x_test, y_test, epochs + 1)  # Train and infer network
            file.write('For {} gradient-descent epochs, the network achieved {:.2f}% classification accuracy, with {:.2f}s training time and {:.2f}s average inference time\n'.format(epochs + 1, acc * 100, train_time, average_inf_time))
        file.write('-----------------------------------------------------------------------------------------------------------------------------\n')
        model = None
        epochs = 10
    if eval_training_size:
        file.write('-------------------------------- Training Sample size Evaluation - | 2 hidden layers | 128 neurons per layer | 10 epochs ----------------------------\n')
        for train_ratio in range(n_nets):
            train_samples = int(len(x_train) * (train_ratio + 1) / 10)
            model = generate_net()
            acc, train_time, average_inf_time = train_and_infer(model, x_train, y_train, x_test, y_test, epochs, train_samples)  # Train and infer network
            file.write('For {} training samples, the network achieved {:.2f}% classification accuracy, with {:.2f}s training time and {:.2f}s average inference time\n'.format(train_samples, acc * 100, train_time, average_inf_time))
        file.write('-----------------------------------------------------------------------------------------------------------------------------\n')
        model = None
    file.close()

def train_and_infer(model, x_train, y_train, x_test, y_test, epochs=10, train_samples=60000):
    """
    Compiles, trains, and infers a given neural network. Returns the accuracy, training time and average inference time of the model
    :param model: Sequential model
    :param x_train: Train data
    :param y_train: Train labels
    :param x_test: Test data
    :param y_test: Test labels
    :param epochs: Number of gradient-descent epochs
    :param train_samples: Number of training samples
    :return: accuracy, training time, average inference time
    """
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    start_time = time()
    model.fit(x_train[:train_samples], y_train[:train_samples], batch_size=32, epochs=epochs, validation_data=(x_test, y_test))
    train_time = time() - start_time
    acc = model.evaluate(x_test, y_test, verbose=1)[1]

    # Evaluate average inference time
    inference_time = 0
    for i in range(1000):
        image = x_test[i]  # Load image
        # image = image.reshape(784)
        image = np.reshape(image, (1, 28, 28))
        start_time = time()
        model.predict(image, batch_size=1)
        inference_time += time() - start_time
    average_inf_time = inference_time / 1000
    model.save('data/trained_net')

    return acc, train_time, average_inf_time
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
