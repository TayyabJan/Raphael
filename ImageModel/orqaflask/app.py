from flask import Flask
import os
from PIL import Image
import joblib
import template as tem

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'




def inference_function(features):
    # model path 
    model_path = 'colour_RFR_model_V1.joblib'
    # load model and predict
    model = joblib.load(model_path)
    print('model is loaded')
    features = features.reshape(1, -1)
    # get prediction of input features
    predict_score = model.predict(features)[0]

    # adjust prediction bound. Prediction is always between 0 and 3
    if predict_score < 0:
        predict_score = 0.0000
    elif predict_score > 3:
        predict_score = 3.0000

    return str(predict_score)


@app.route('/model')
def model():
    print("model endpoint is hit")
        # liver_colour_models
    
    if __name__ == '__main__':

        # image path
        file = 'image59.jpg'

        # check if the input image is exist
        if not os.path.isfile(file):
            print(file, ' Not exist')
            exit()
        # Extract texture features
        liver_img = Image.open(file)
        # extract features
        features_ = tem.feature_extraction(liver_img)
        if len(features_) == 0:
            print('There are not features')
            exit()

        prediction = inference_function(features_)
        print('The predicted score for the input image is: ', prediction)

    return 'Yes Prediction Made Score is ' + prediction

if __name__ == '__main__':
    app.run()


