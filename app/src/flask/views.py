from flask import Flask, render_template, request
from app.src.classifier import classification_handler as ch

app = Flask(__name__, template_folder='../templates')
model = ch.load_fmnist_model()
classes = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

@app.route('/', methods=['GET', 'POST'])
def home():
    image_path = None
    msg = None
    if request.method == 'POST':
        image = request.files['input_image']
        image.save('data/images/{}'.format(image.filename))
        image_path = 'data/images/{}'.format(image.filename)
        if image_path:
            processed_img = ch.process_image(image_path)
            prediction = model.predict(processed_img, batch_size=1)
            prediction = prediction.tolist()
            msg = classes[prediction[0].index(max(prediction[0]))]
    return render_template('home.html', msg=msg)
