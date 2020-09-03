from app.src.classifier.mnist_classifier import *

# -------------------------------------------------------- Main ---------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()
    # print_images(x_train)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # # Initialize and train model
    # model = net_init()
    # model.summary()
    # model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    # model.evaluate(x_test, y_test, verbose=2)

    evaluate_nets(x_train, y_train, x_test, y_test, eval_layers=True, eval_neurons=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
