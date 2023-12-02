from flask import Flask,request, jsonify,render_template
import base64
import os
from flask_cors import CORS
import io
import json
import os
import cv2
import numpy as np
import glob
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import math

app = Flask(__name__, static_url_path = "/tmp", static_folder = "tmp")
CORS(app)


# Define categories
categories = ['Raphael', 'Not Raphael']

ResNet_Path = "models/resnet50_model.h5"
Model_Path = "models/28_09_2023_svm_final_model.pkl"

Raphael_Prob = 0.0
Not_Raphael_Prob = 0.0

def scale_inverse_log(x, x_min, x_max, y_min, y_max):
    # Check input boundaries
    if x < x_min or x > x_max:
        return "Input x must be within the range [x_min, x_max]"

    # Calculate inverse log of x
    inv_log_x = -1 / math.log(x + 1)

    # Calculate inverse log of x_min and x_max
    inv_log_x_min = -1 / math.log(x_min + 1)
    inv_log_x_max = -1 / math.log(x_max + 1)

    # Scale the inverse logarithmic value to the target range [y_min, y_max]
    y = y_min + (inv_log_x - inv_log_x_min) * (y_max - y_min) / (inv_log_x_max - inv_log_x_min)

    return y


# Function to load and preprocess image
def load_and_preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def extract_features(img_path, model):
    img = load_and_preprocess_image(img_path)
    features = model.predict(img)
    return features.reshape(-1)

def visualize_probabilities(test_image_path, categories, probabilities):
    # Load image
    # test_image_path = "./Tests/2016 Discovery Innocenzo Francucci da Imola The Virgin.jpg"
    img = Image.open(test_image_path)
    print(test_image_path)
    # Create subplots
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot image on first subplot
    # ax1.imshow(img)
    # ax1.axis('off')
    # plt.show()

    # Create a bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(categories, probabilities)
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Probabilities for each class')

    print("test image path is " , test_image_path)
    File = os.path.basename(test_image_path)
    print("The FILE IS", File)

    # file_name = result[len(result)-1]

    print("The FILE IS  *********************", File)
    plt.savefig('uploads/bar_'+ str(File))
    
    # plt.show()

# Function to calculate edge features using Canny edge detector
def calculate_canny_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.std(edges)

# Function to calculate edge features using Sobel operator
def calculate_sobel_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return np.std(sobelx), np.std(sobely)

# Function to calculate edge features using Laplacian operator
def calculate_laplacian_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.std(laplacian)

# Function to calculate edge features using Scharr operator
def calculate_scharr_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    return np.std(scharrx), np.std(scharry)

# Function to calculate all edge features
def calculate_features(img):
    canny_edges = calculate_canny_edges(img)
    sobel_edges_x, sobel_edges_y = calculate_sobel_edges(img)
    laplacian_edges = calculate_laplacian_edges(img)
    scharr_edges_x, scharr_edges_y = calculate_scharr_edges(img)
    return np.array([canny_edges, sobel_edges_x, sobel_edges_y, laplacian_edges, scharr_edges_x, scharr_edges_y])

def compare_image_with_dataset(test_image_path, image_dir, categories):


    # Load test image
    test_image = cv2.imread(test_image_path)


    # Load the final model
    svm_final = joblib.load(Model_Path)

    # Load the saved model
    model = load_model(ResNet_Path)

    # Extract features from the test image
    test_image_features = extract_features(test_image_path, model)

    # Use the loaded model to predict the category of the test image
    predicted_category = svm_final.predict([test_image_features])[0]

    # Calculate probabilities for each category
    probabilities = svm_final.predict_proba([test_image_features])[0]

    categories = ['Raphael', 'Not Raphael']

    # Calculate features of test image
    test_features = calculate_features(test_image)

    # Normalize test features to get weights
    weights = test_features / np.sum(test_features)

    # Load all images in directory
    formats = ('*.jpg', '*.png', '*.bmp')  # Add or remove formats as needed
    image_paths = []
    for fmt in formats:
        image_paths.extend(glob.glob(f"{image_dir}/{fmt}"))

    # Calculate the total feature values and the count of images
    total_features = np.zeros_like(test_features)
    image_count = 0

    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path)

        # Calculate features of image
        image_features = calculate_features(image)

        # Add to total and increment count (multiply by weights here)
        total_features += image_features * weights
        image_count += 1

    # Calculate the weighted average feature values
    average_features = total_features / image_count if image_count else np.zeros_like(test_features)

    # Compare average features with test image
    difference = np.abs(test_features - average_features)

    # Sum of differences
    sum_diff = np.sum(difference)

    max_diff = np.max(difference)
    min_diff = np.min(difference)
    mean_diff = np.mean(difference)
    edge_threshold = (mean_diff - min_diff) / mean_diff
    



    #penalise for edge information (these values can be found by experimentation)
    if mean_diff < 50:
        mean_diff = 400
        probabilities[0] = probabilities[0] - 0.3

    if mean_diff > 400:
        mean_diff = 400
        probabilities[0] = probabilities[0] - 0.3

    if mean_diff < 150:
        mean_diff = 150

    scale_ = scale_inverse_log(mean_diff, x_min=150, x_max=400, y_min=0.0, y_max=-0.99)
    print (scale_)

    threshold = 0.95* probabilities[0] + 0.05*scale_
    if threshold < 0:
       threshold = 0.05

    probabilities[0] = threshold
    probabilities[1] = 1 - threshold
    # print("Probabilities", probabilities[0], probabilities[1])
    Raphael_Prob = probabilities[0]
    print("Raphael_Prob: ", Raphael_Prob)
    Not_Raphael_Prob = probabilities[1]
    print("Not_Raphael_Prob: ", Not_Raphael_Prob)
    # visualize_probabilities(test_image_path ,categories, probabilities)
    return(probabilities[0], probabilities[1])



@app.route('/')
def maincall():
    # return {'Message' : 'This is abc'}
    # return {'Message:': 'Welcome to Raphael Web'}
    return render_template("index.html")

@app.route('/css')
def css():
    # return {'Message' : 'This is abc'}
    # return {'Message:': 'Welcome to Raphael Web'}
    return render_template("orqa_styles.css")



@app.route('/modeltest',methods=['GET','POST'])
def modelcall():
    # return {'Message' : 'This is abc'}
    print("modeltest is hit")
    return json.dumps({'message': 'There you go............ '})



@app.route('/model')
def model():

    test_image_dir = f"./Tests"
    image_dir = f"./Raphael"
    
    #Load the trained model and Resnet base model
    

    # # Define directory containing test images and training images (for edge feature extraction)
   
    if os.path.exists(image_dir):
        print("path exists: ")
    else:
        return

    # Get all test images in directory
    formats = ('*.jpg', '*.png', '*.bmp')
    test_image_paths = []
    for fmt in formats:
        # test_image_paths.extend(glob.glob(f"{test_image_dir}/{fmt}"))
        test_image_paths.extend(glob.glob(os.path.join(test_image_dir, fmt)))
    test_image_paths = [path.replace("\\", "/") for path in test_image_paths]
    # print("The image pATHS ", test_image_paths)
    # print("The image array length is ", len(test_image_paths))
    # return
    # Loop through each test image and compare with dataset
    
    for test_image_path in test_image_paths:
        print("test_image_path: ", test_image_path)
        a,b = compare_image_with_dataset(test_image_path, image_dir, categories)
        # print("probabilities[0], probabilities[1]: ", a,b) 
        Raphael_Prob = a
        Not_Raphael_Prob = b 
        break
    # print("Yes the model is executed successfully...........")
    return {'Raphael': str(Raphael_Prob),
             'Not_Raphael': str(Not_Raphael_Prob)   }
    # return {'Message': abc }



UPLOAD_FOLDER = 'uploads'

#***********************************************
@app.route('/submit_form', methods=['POST'])
def submit_form():
    # render_template("test2.html")
    if 'image' in request.files:
        file = request.files['image']
        print("Uploaded image name: ", file)
        file_path = os.path.join(UPLOAD_FOLDER, 'abc.' + str("png"))
        saveFile(file,file_path)
        print("Uploaded file_path image name and path: ", file_path)

        Raphael_Prob,Not_Raphael_Prob = compare_image_with_dataset(file_path, f"./Raphael", categories)
            
        return {'message': 'Processing model result',
                    'Raphael': str(Raphael_Prob),
                'Not_Raphael': str(Not_Raphael_Prob)
            }
    # return render_template("test.html")
    # return jsonify({'message': 'Form Submitted Successfully', 'file_path': file_path})
#*****************************************



#***********************************************

@app.route('/submit_form2', methods=['POST'])
def submit_form2():
    name = request.form['name']
    print("Second Call", name)
    return 'Form Submitted Successfully'



#*****************************************


def saveFile(file,path):
    try:
        file.save(path)
    except ValueError as ve:
        print("File Error")


@app.route('/upload1', methods=['GET','POST'])
# def upload1():
#     try:
#         data = request.json
        
#         File_Name = ""
#         base64string = ""


#         for key, value in data.items():
#              if key == 'file_extension':
#                  File_Name = value
#              else:
#                  base64string = value
             
        
#         # base64string = request.form['image']
#         # base64string = data.image
#         # # # print("upload is hit", base64string)
#         # # File_Name = request.form['file_extension']
#         # File_Name = data.file_extension
#         file_path = os.path.join(UPLOAD_FOLDER, 'image.' + str(File_Name))

#         # # Decode the base64 string
#         imgdata = base64.b64decode(base64string)

#         # # Open the image using PIL
#         img = Image.open(io.BytesIO(imgdata))

#         # # Save the image to a file
#         saveFile(img,file_path)
#         print(file_path)
        
#         return {'message': 'Image Processed Successfully'}
#     except ValueError as ve:
#         # return {'message': 'It seems some issue occured. error detail is :' + ve }
#         return {'message': 'It seems some issue occured. error detail is :' + ve }
def upload1():
    if 'image' in request.files:
        file = request.files['image']
        print(file)
        file_path = os.path.join(UPLOAD_FOLDER, 'abc.' + str("png"))
        saveFile(file,file_path)
        # Now you can save the file or process it as you wish
    return 'File uploaded successfully'
@app.route('/upload', methods=['GET','POST'])
def upload():
    try:
        # base64string = request.form['image']
        # # print("upload is hit", base64string)
        # File_Name = request.form['file_extension']
        # file_path = os.path.join(UPLOAD_FOLDER, 'image.' + str(File_Name))

        # # Decode the base64 string
        # imgdata = base64.b64decode(base64string)

        # # Open the image using PIL
        # img = Image.open(io.BytesIO(imgdata))

        # # Save the image to a file
        # saveFile(img,file_path)
        # # print(file_path)

        if 'image' in request.files:
            file = request.files['image']
            print("Uploaded image name: ", file)
            file_path = os.path.join(UPLOAD_FOLDER, 'abc.' + str("png"))
            saveFile(file,file_path)
            print("Uploaded file_path image name and path: ", file_path)

            Raphael_Prob,Not_Raphael_Prob = compare_image_with_dataset(file_path, f"./Raphael", categories)
                
            return {'message': 'Processing model result',
                        'Raphael': str(Raphael_Prob),
                    'Not_Raphael': str(Not_Raphael_Prob)
                }
        
    except ValueError as ve:
        # return {'message': 'It seems some issue occured. error detail is :' + ve }
        return {'message': 'It seems some issue occured. error detail is :' + ve }



@app.route('/process', methods=['GET','POST'])
def process():
    try:
        test_image_dir = f"./Tests"
        image_dir = f"./Raphael"
        
        file_extension = request.form['file_extension']
        # print("process is hit file name is ", base64string)
        test_image_path = os.path.join(UPLOAD_FOLDER, 'image.' + str(file_extension))
        # print(test_image_path)
        # if os.path.exists(test_image_path):
        #     print("Yes path exists")
        # else:
        #     print("Path does not exists")

        Raphael_Prob,Not_Raphael_Prob = compare_image_with_dataset(test_image_path, image_dir, categories)
        
        return {'message': 'Processing model result',
                    'Raphael': str(Raphael_Prob),
                'Not_Raphael': str(Not_Raphael_Prob)
            }
    except ValueError as ve:
            # return {'message': 'It seems some issue occured. error detail is :' + ve }
            return {'message': 'It seems some issue occured. error detail is :' + str(ve) }



if __name__ == '__main__':
    app.run()


