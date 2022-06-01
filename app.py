from flask import Flask, request
from fastai.basics import *
from fastai.vision.all import *
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)
learn = load_learner('trained_model.pkl')


def predict_image(img):
    prediction = learn.predict(PILImage.create(img))
    if prediction[0] == 'PNEUMONIA':
        return 'PNEUMONIA'
    return 'NORMAL'


@app.route('/predict', methods=['POST'])
def predict():
    return predict_image(request.files['image'])


if __name__ == '__main__':
    app.run