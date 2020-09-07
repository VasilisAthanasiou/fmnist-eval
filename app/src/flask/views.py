from flask import Flask, render_template, request, session
from app.src.classifier import classification_handler as ch
import numpy as np
import os

app = Flask(__name__, template_folder='../templates', static_folder='../../data/static')
model = ch.load_fmnist_model(model='cnn')
classes = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
class_desc = ['T-Shirts cost 10 euros', 'Buy one trouser, get another one for free', 'Pullovers are made of the finest materials', 'Dresses are 50% off', 'Coats are out of stock', "Sandal's only available size is 45", 'Shirts are shirts', 'Sneakers make no sound', 'Bags are used to store items inside them', 'Ankle boots are like boots, but for ankles']
ranked_images = ch.rank_images()

@app.route('/', methods=['GET', 'POST'])
def home():
    image_name = None
    top_images_name = None
    category = None
    description = None
    is_processed = False

    if request.method == 'POST':
        image = request.files['input_image']
        image_name = image.filename
        image.save('data/static/{}'.format(image_name))
        image_path = 'data/static/{}'.format(image_name)
        if image_path:

            if 'is_processed' in request.form:
                is_processed = int(request.form['is_processed'])
            processed_img = ch.process_image(image_path, )
            prediction = model.predict(processed_img, batch_size=1)
            prediction = prediction.tolist()
            class_index = prediction[0].index(max(prediction[0]))
            category = classes[class_index]
            description = class_desc[class_index]

            # Get top n images of predicted class
            post_fix = np.random.randint(10000)
            top_images = ch.top_k_images(ranked_images, class_index, n=int(request.form['k_top']))

            # Remove previous top-images so that flask will update the displayed image
            [os.remove(os.path.join('data/static', f)) for f in os.listdir('data/static') if f.startswith('top-images')]

            # Save top images to local directory
            ch.save_images(top_images, 'top-images{}.png'.format(post_fix))
            top_images_name = 'top-images{}.png'.format(post_fix)

    return render_template('home.html', category=category, description=description, image_name=image_name, top_images_name=top_images_name)

