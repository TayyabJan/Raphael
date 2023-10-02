from flask import Flask,request, jsonify,render_template
import base64
import os
from flask_cors import CORS
from PIL import Image
import joblib
import template as tem
import json
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/abc')
def hello():
    score = 1.5
    # return {'Message' : 'This is abc'}
    return render_template('orqaresults.html',score=score)


def inference_function(features):
    model_path = 'colour_RFR_model.joblib'
    model = joblib.load(model_path)
    features = features.reshape(1, -1)
    # get prediction of input features
    predict_score = model.predict(features)[0]
   
    # adjust prediction bound. Prediction is always between 0 and 3
    if predict_score < 0:
        predict_score = 0.0000
    elif predict_score > 3:
        predict_score = 3.0000
    label = ''
    predict_score += 0.5
    if predict_score < 1:
        label = 'None'
    elif 1 < predict_score < 2:
        label = 'Mild'
    elif 2 <= predict_score < 3.0:
        label = 'Moderate'
    elif predict_score >= 3.0:
        label = 'Severe'

    return predict_score, label

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
        modelResults =    getModelResults(file_path)
        print("The model Results are ", modelResults)
        # return {'message': modelScore}
        # return render_template('orqaresults.html', score = modelScore)
        return   json.dumps(modelResults, default=convert_to_json_serializable)
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

        prediction, pred_label = inference_function(features_)
        # print('The predicted image score for the input image is: ', prediction)

        return  {'message':round(prediction,2),'label': pred_label,'Red': features_[0], 'Green':features_[1], 'Blue': features_[2]}


# Convert int32 values to regular Python integers
def convert_to_json_serializable(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    return obj

@app.route('/model')
def model():
    print("model endpoint is hit")
        # liver_colour_models
    
    if __name__ == '__main__':

        # image path
        file = 'image59.jpg'
    modelResults =    getModelResults(file)
    print(type(modelResults))
    # return str(modelResults)
    return json.dumps(modelResults, default=convert_to_json_serializable)
    # return 'Yes Im    age Prediction Made Score is ' + str(prediction) + ' label is :' + pred_label

if __name__ == '__main__':
    app.run()


