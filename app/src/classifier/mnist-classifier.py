import tensorflow as tf
from app.src.classifier import mnist_reader
from matplotlib import pyplot as plt

# ------------------------------------------ Dataset operations ---------------------------------------------------------------------------------------------- #
def load_dataset(path='../../data'):
    xtrain, ytrain = mnist_reader.load_mnist(path, kind='train')  # Extract f-mnist data using load_mnist from the mnist repository
    xtest, ytest = mnist_reader.load_mnist(path, kind='t10k')  # https://github.com/zalandoresearch/fashion-mnist

    return xtrain, ytrain, xtest, ytest

def print_images(images):
    for i in range(9):
        plt.subplot(331 + i)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))
    plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------------- Main ---------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()
    print_images(x_train)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
