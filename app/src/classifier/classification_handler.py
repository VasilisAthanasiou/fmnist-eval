from app.src.classifier.mnist_classifier import *
import cv2 as cv

# -------------------------------------------------------- Train ---------------------------------------------------------------------------------------------- #

def init_and_train(x_train, y_train, x_test, y_test):

    # # Initialize and train model
    # model = net_init()
    # model.summary()
    # model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    # model.evaluate(x_test, y_test, verbose=2)

    # evaluate_nets(x_train, y_train, x_test, y_test, n_nets=20, eval_epochs=True)
    model = net_init()
    model.summary()
    acc, train_time, avg_inf_time = train_and_infer(model, x_train, y_train, x_test, y_test, epochs=19)
    print("{:.2f}, {:.2f}".format(acc, train_time))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

def load_fmnist_model():
    return tf.keras.models.load_model('data/trained_net')

def process_image(path):
    print(path)
    if path:
        image = cv.imread(path)
        # Convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Resize to 28x28
        resized = cv.resize(gray, (28, 28))
        reshaped = np.reshape(resized, (1, 28, 28))
        reshaped = (255-reshaped)

        return reshaped
    return None
