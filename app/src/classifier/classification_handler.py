from app.src.classifier.mnist_classifier import *
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

# -------------------------------------------------------- Train ---------------------------------------------------------------------------------------------- #

def init_and_train(x_train, y_train, x_test, y_test):

    model = net_init()
    model.summary()
    acc, train_time, avg_inf_time = train_and_infer(model, x_train, y_train, x_test, y_test, epochs=19)
    print("{:.2f}, {:.2f}".format(acc, train_time))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

def load_fmnist_model(model='classic'):
    if model == 'classic':
        return tf.keras.models.load_model('data/trained_net')
    elif model == 'cnn':
        return tf.keras.models.load_model('data/trained_cnn')

def process_image(path, process=True):
    print(path)
    resized = None
    if path:
        image = cv.imread(path)
        if process:
            # Convert to grayscale
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # Resize to 28x28
            resized = cv.resize(gray, (28, 28))
            result = (255-resized)
            # result = np.reshape(resized, (1, 28, 28))
        try:
            result = np.reshape(resized, (1, 28, 28, 1))
        except Exception:
            return image
        # result = result / 255
        return result

    return None

def rank_images():
    images, labels, _, _ = load_dataset(normalize=False)
    scores = np.random.randint(101, size=len(images))  # Assign random scores for images from 0 to 100
    data = list(zip(labels, images, scores))
    data = sorted(data, key=lambda x: x[2])

    # Initialize map
    img_map = []
    for i in range(10):
        img_map.append([])

    for elem in data:
        img_map[elem[0]].append(elem)

    return img_map

def top_k_images(img_map, category, n=5):
    res = []
    for i in range(1, n + 1):
        res.append(img_map[category][-i][1])
    return res

def save_images(images, name):

    for i in range(len(images)):
        plt.subplot(331 + i)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))
    plt.savefig('data/static/{}'.format(name))
    plt.close()


