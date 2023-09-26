import numpy as np
from numpy import asarray, log10, clip
import random
from sklearn.ensemble import RandomForestRegressor

random.seed(0)
'''
model_info
purpose: required to specify the type of model, the images to use and the data to record for the model run
'''
# model_info = {
#    "model_name" = "template",
#    "organ" = "liver",
#    "dataset" = "NORIS",
#    "type" = "prediction", # prediction or classification
#    "predicted_variable" = "steatosis"
# }


'''
function: image_processing
purpose: to perform any preprocessing on the images, e.g. background removal, prior to the feature extraction 
input: PIL Image from NORIS dataset
output: PIL Image
application: this function is applied to all images in the dataset as a preprocessing step
'''


def image_processing(image):
    # This example does no image processing - just returns the original image
    return image


'''
function: feature_extraction
purpose: to extract features from the images, e.g. colour or texture, to be used in the ML model
input: PIL Image (output from the image_processing)
output: numpy ndarray with 1 row and columns representing the features
application: this function is applied to all images in the dataset following the preprocessing step 
and the output is combined into a numpy ndarray with columns representing the features and rows representing the images
'''


def collect_data(ycoordinatepixels, xcoordinatepixels, data, hlen, wlen):
    count = 0
    red1 = 0
    green1 = 0
    blue1 = 0

    for i in range(ycoordinatepixels, ycoordinatepixels + hlen, 3):
        for j in range(xcoordinatepixels, xcoordinatepixels + wlen, 3):
            count = count + 1
            red1 = red1 + (data[i, j][0] ** 2)
            green1 = green1 + (data[i, j][1] ** 2)
            blue1 = blue1 + (data[i, j][2] ** 2)

    red1 = int((red1 / count) ** 0.5)
    green1 = int((green1 / count) ** 0.5)
    blue1 = int((blue1 / count)) ** 0.5

    red1 = clip(red1, 1, 255)
    green1 = clip(green1, 1, 255)
    blue1 = clip(blue1, 1, 255)

    return red1, green1, blue1


def is_background(y, x, data):  # improvement on ==[0,0,0,0]).all()
    is_bg = False

    red = data[y, x][0]
    green = data[y, x][1]
    blue = data[y, x][2]

    red = clip(red, 1, 255)
    green = clip(green, 1, 255)
    blue = clip(blue, 1, 255)

    luminosity = (red + green + blue) / 3

    if luminosity < 40:  # too dark
        is_bg = True

    if luminosity > 167:  # too light
        is_bg = True

    if luminosity < 60 and red / (green + blue) > 2:  # too red
        is_bg = True

    if luminosity >= 60 and red / (green + blue) > 1.25:  # too red
        is_bg = True

    if blue / luminosity > 1.2:  # too blue
        is_bg = True

    # new saturation addition
    temp = luminosity / 3
    saturation = (abs(red - temp) + abs(green - temp) + abs(blue - temp)) / temp

    if saturation < 0.7:
        is_bg = True

    return is_bg


def is_not_liver(luminosity, redgreen, blueyellow, minluminosity, maxluminosity, minredgreen, maxredgreen,
                 minblueyellow, maxblueyellow):
    is_not_liver = False

    if luminosity < minluminosity or luminosity > maxluminosity:
        is_not_liver = True

    if redgreen < minredgreen or redgreen > maxredgreen:
        is_not_liver = True

    if blueyellow < minblueyellow or blueyellow > maxblueyellow:
        is_not_liver = True

    return is_not_liver


def stats(variable, array):
    # Welford
    # #array[0] = number
    # array[2] = mean
    # stdev = (array[4]/array[0]) ** 0.5
    array[0] = array[0] + 1
    array[1] = variable - array[2]
    array[2] = array[2] + (array[1] / array[0])
    array[3] = variable - array[2]
    array[4] = array[4] + (array[1] * array[3])

    return array


def feature_extraction(Starting_liver_img):
    red2_s, green2_s, blue2_s, luminosity_s, redgreen_s, blueyellow_s = [], [], [], [], [], []

    # Convert from RGB to RGBA so code is compatible with rembg
    Starting_liver_img = Starting_liver_img.convert("RGBA")
    data = asarray(Starting_liver_img)  # read only image data
    data_modified = data.copy()  # modified pixel data for screen display etc

    # Image sampling variables

    uboundy = int(data_modified.shape[0] * 0.03)  # y coordinate of image top
    lboundy = int(data_modified.shape[0] * 0.97)  # y coordinate of image bottom
    lboundx = int(data_modified.shape[1] * 0.03)  # x coordinate of image left
    rboundx = int(data_modified.shape[1] * 0.97)  # x coordinate of image right

    wlen = int(data_modified.shape[1] / 200)  # width of sample square in pixels
    hlen = int(data_modified.shape[0] / 200)  # height of sample square in pixels

    minx = 40000
    miny = 40000
    maxx = 0
    maxy = 0

    luminosityarray = np.zeros([5])
    redgreenarray = np.zeros([5])
    blueyellowarray = np.zeros([5])

    # First pass to find approximate luminosity, red-green and blue-yellow values for this liver

    CIELABLumTotal = 0
    CIELABLumNumber = 0
    CIELABLumAverage = 0.0
    CIELABBYTotal = 0
    CIELABBYNumber = 0
    CIELABBYAverage = 0.0

    for k in range(1, 200):

        is_initial_backgroundestimate = False
        CIELABLum = 0
        CIELABBY = 0

        xcoordinatepixels = random.randint(lboundx, rboundx)
        ycoordiatepixels = random.randint(uboundy, lboundy)
        red1, green1, blue1 = collect_data(ycoordiatepixels, xcoordinatepixels, data, hlen, wlen)

        luminosity = red1 + green1 + blue1
        redgreen = int(log10((green1 + blue1) / (2 * red1)) * 1000)
        blueyellow = int(log10((2 * blue1) / (red1 + green1)) * 1000)
        temp = luminosity / 3
        saturation = (abs(red1 - temp) + abs(green1 - temp) + abs(blue1 - temp)) / temp

        if luminosity < 120 or luminosity > 501:
            is_initial_backgroundestimate = True

        if blueyellow < -800 or blueyellow > -50:
            is_initial_backgroundestimate = True

        if redgreen < -450 or redgreen > -50:
            is_initial_backgroundestimate = True

        if saturation < 0.7:
            is_initial_backgroundestimate = True

        if luminosity < 180 and red1 / (green1 + blue1) > 2:
            is_initial_backgroundestimate = True

        if luminosity >= 180 and red1 / (green1 + blue1) > 1.25:
            is_initial_backgroundestimate = True

        if blue1 / luminosity > 0.4:
            is_initial_backgroundestimate = True

        if is_initial_backgroundestimate == False:

            if luminosity >= 150 and luminosity <= 233:
                CIELABLum = 1
            elif luminosity >= 234 and luminosity <= 316:
                CIELABLum = 2
            elif luminosity >= 317 and luminosity <= 401:
                CIELABLum = 3
            else:
                CIELABLum = 0

            if blueyellow > -300 and blueyellow < -50:
                CIELABBY = 1
            elif blueyellow > -550 and blueyellow < -301:
                CIELABBY = 2
            elif blueyellow > -800 and blueyellow < -551:
                CIELABBY = 3
            else:
                CIELABBY = 0

            if CIELABLum > 0 and CIELABBY > 0:
                CIELABLumTotal = CIELABLumTotal + CIELABLum
                CIELABLumNumber = CIELABLumNumber + 1
                CIELABBYTotal = CIELABBYTotal + CIELABBY
                CIELABBYNumber = CIELABBYNumber + 1

    if CIELABLumNumber > 0:
        CIELABLumAverage = CIELABLumTotal / CIELABLumNumber
    else:
        print('CIELABLumNumber = 0')

    if CIELABBYNumber > 0:
        CIELABBYAverage = CIELABBYTotal / CIELABBYNumber
    else:
        print('CIELABBYNumber = 0')

    # Test variation as a measure of liver quality

    # Initial luminosity, red - green  and blue - yellow values for 'all liver' colour-space
    # See 5-6-23 Validate the new bg removal prescreen

    minluminosity = int((CIELABLumAverage * 86) + 24)
    maxluminosity = int((CIELABLumAverage * 86) + 192)
    minredgreen = -450  # -250
    maxredgreen = -50
    minblueyellow = int((-250 * CIELABBYAverage) - 175)
    maxblueyellow = int((-250 * CIELABBYAverage) + 325)

    if minluminosity < 150:
        minluminosity = 150

    if maxluminosity > 400:
        maxluminosity = 400

    if maxblueyellow > -50:
        maxblueyellow = -50

    if minblueyellow < -800:
        minblueyellow = -800

    stdev = 0.0

    # Evaluation variables
    red2 = 0
    green2 = 0
    blue2 = 0
    count2 = 0

    luminosity, redgreen, blueyellow = 0, 0, 0
    for pas in range(1, 6):

        if pas > 1:
            lboundx = minx
            rboundx = maxx
            uboundy = miny
            lboundy = maxy

            stdev = (luminosityarray[4] / luminosityarray[0]) ** 0.5
            minluminosity = luminosityarray[2] - stdev
            maxluminosity = luminosityarray[2] + stdev

            # if pas == 4:
            # print('stdev luminosity = ' + str(stdev))

            stdev = (redgreenarray[4] / redgreenarray[0]) ** 0.5
            minredgreen = redgreenarray[2] - stdev
            maxredgreen = redgreenarray[2] + stdev

            # if pas == 4:
            # print('stdev redgreen = ' + str(stdev))

            stdev = (blueyellowarray[4] / blueyellowarray[0]) ** 0.5
            minblueyellow = blueyellowarray[2] - stdev
            maxblueyellow = blueyellowarray[2] + stdev

            # if pas == 4:
            # print('stdev blueyellow = ' + str(stdev))

        if pas == 5:
            luminosityarray[:] = 0
            redgreenarray[:] = 0
            blueyellowarray[:] = 0

        for k in range(1, 200):

            xcoordinatepixels = random.randint(lboundx, rboundx)
            ycoordiatepixels = random.randint(uboundy, lboundy)

            if is_background(ycoordiatepixels, xcoordinatepixels,
                             data):  # Look at first random pixel. Is it background?
                data_modified[ycoordiatepixels:ycoordiatepixels + hlen, xcoordinatepixels:xcoordinatepixels + hlen] = \
                    [0, 0, 255, 255]  # If bg try again
                # print('miss -  not foreground')
            else:
                red1, green1, blue1 = collect_data(ycoordiatepixels, xcoordinatepixels, data, hlen,
                                                   wlen)  # if foreground measure RMS RGB for sample square
                luminosity = (red1 + green1 + blue1)  # calculate luminosity, R-G and B-Y values
                redgreen = int(log10((green1 + blue1) / (2 * red1)) * 1000)
                blueyellow = int(log10((2 * blue1) / (red1 + green1)) * 1000)

                if is_not_liver(luminosity, redgreen, blueyellow, minluminosity, maxluminosity, minredgreen,
                                maxredgreen, minblueyellow, maxblueyellow):  # Is it liver?
                    data_modified[ycoordiatepixels:ycoordiatepixels + hlen,
                    xcoordinatepixels:xcoordinatepixels + hlen] = [
                        255, 0, 0, 255]
                    # print("miss - not liver")
                else:
                    data_modified[ycoordiatepixels:ycoordiatepixels + hlen,
                    xcoordinatepixels:xcoordinatepixels + hlen] = [
                        0, (50 * pas), 0, 255]
                    # print("pas = " + str(pas) + " k = " + str(k) + " red1 = " + str(red1) + " green1 = " + str(green1) + " blue1 = " + str(blue1))

                    luminosityarray = stats(luminosity, luminosityarray)
                    redgreenarray = stats(redgreen, redgreenarray)
                    blueyellowarray = stats(blueyellow, blueyellowarray)

                    if pas == 5:
                        red2 = red2 + red1
                        red2_s.append(red2)
                        green2 = green2 + green1
                        green2_s.append(green2)
                        blue2 = blue2 + blue1
                        blue2_s.append(blue2)
                        count2 = count2 + 1
                        luminosity = (red1 + green1 + blue1)  # calculate luminosity, R-G and B-Y values
                        luminosity_s.append(luminosity)
                        redgreen = int(log10((green1 + blue1) / (2 * red1)) * 1000)
                        redgreen_s.append(redgreen)
                        blueyellow = int(log10((2 * blue1) / (red1 + green1)) * 1000)
                        blueyellow_s.append(blueyellow)

                minx = min(minx, xcoordinatepixels)
                miny = min(miny, ycoordiatepixels)
                maxx = max(maxx, xcoordinatepixels)
                maxy = max(maxy, ycoordiatepixels)
    if len(red2_s) == 0 or len(green2_s) == 0 or len(blue2_s) == 0 or len(luminosity_s) == 0 or len(redgreen_s) == 0 \
            or len(blueyellow_s) == 0:
        return None
    red2_s = np.max(red2_s)
    green2_s = np.max(green2_s)
    blue2_s = np.max(blue2_s)
    luminosity_s = np.max(luminosity_s)
    redgreen_s = np.max(redgreen_s)
    blueyellow_s = np.max(blueyellow_s)
    return np.asarray([red2_s, green2_s, blue2_s, luminosity_s, redgreen_s, blueyellow_s])


'''
function: train_model
purpose: to train the model used to predict the Surgeons' score
input: numpy ndarray with columns representing the features and rows representing the images
output: model - to be saved as pickle or joblib file
application: this function is used to train the ML model to predict the surgeon's score based on the training data 
'''


def train_model(features_train, scores_train):
    # Random Forest Repression to predict the score based on the features
    model = RandomForestRegressor()

    model.fit(features_train, scores_train)
    return model


'''
function: test_model
purpose: to apply the model to a new dataset (not used for training) e.g. for testing, validation or inference
input: features - numpy ndarray with columns representing features and rows representing the images in the dataset
       model - pickle or joblib file containing the model
output: model predictions for the images in the test dataset
application: this function is executed following the training to assess the model performance on the test dataset
'''


def test_model(features_test, model):
    scores = model.predict(features_test)
    return scores

