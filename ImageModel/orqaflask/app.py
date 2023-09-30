from flask import Flask,request, jsonify,render_template
import base64
import os
from flask_cors import CORS
from PIL import Image
import joblib
import template as tem

app = Flask(__name__)
CORS(app)

@app.route('/abc')
def hello():
    score = 1.5
    # return {'Message' : 'This is abc'}
    return render_template('orqaresults.html',score=score)


def inference_function(features):
    # model path 
    model_path = 'colour_RFR_model.joblib'
    # load model and predict
    model = joblib.load(model_path)
    # print('model is loaded')
    features = features.reshape(1, -1)
    # get prediction of input features
    predict_score = model.predict(features)[0]

    # adjust prediction bound. Prediction is always between 0 and 3
    if predict_score < 0:
        predict_score = 0.0000
    elif predict_score > 3:
        predict_score = 3.0000

    return str(predict_score)

UPLOAD_FOLDER = 'uploads'


def saveFile(file,path):
    try:
        file.save(path)
    except ValueError as ve:
        print("File Error")

    

@app.route('/upload', methods=['GET','POST'])
def upload():
    try:
        file = request.files['image']
        file_Name = request.form['filename']
        file_path = os.path.join(UPLOAD_FOLDER, file_Name)
        print(file_path)
        saveFile(file,file_path)
        print("upload is hit")
        # # modelScore = 1.456
        # modelScore =    getModelResults(file_path)
        # print("The model score is ", modelScore)
        # return {'message': modelScore}
        return {'message': 'Image Processed Successfully :'}
    except ValueError as ve:
        # return {'message': 'It seems some issue occured. error detail is :' + ve }
        return {'message': 'It seems some issue occured. error detail is :' }


@app.route('/process', methods=['GET','POST'])
def process():
    try:
         
        file_Name = request.form['filename']
        print("process is hit file name is ", file_Name)
        file_path = os.path.join(UPLOAD_FOLDER, file_Name)
        print(file_path)
        modelScore =    getModelResults(file_path)
        print("The model score is ", modelScore)
        # return {'message': modelScore}
        # return render_template('orqaresults.html', score = modelScore)
        return {'message': modelScore}
    except ValueError as ve:
        # return {'message': 'It seems some issue occured. error detail is :' + ve }
        return {'message': 'It seems some issue occured. error detail is :' }


def getModelResults (file_path):
        # print("Model function is called")
    # image path
        # file = 'image59.jpg'
        file = file_path
        # print("File path is   " ,file)
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
        # print('The predicted image score for the input image is: ', prediction)

        return  prediction


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
        print('The predicted image score for the input image is: ', prediction)

    return 'Yes Image Prediction Made Score is ' + prediction

if __name__ == '__main__':
    app.run()


