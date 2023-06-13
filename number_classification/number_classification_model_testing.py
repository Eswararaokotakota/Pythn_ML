import cv2
import numpy as np
import tensorflow as tf
import sys
import glob
import os
print(tf.__version__)

from tensorflow.keras.models import load_model


path1 = r"C:\Users\oculo\Desktop\Eswar\Projects\number_classification\test_data"
files=glob.glob(path1+"\*")
files.sort()
for i in files[:]:
    name = os.path.split(i)[1]
    image = cv2.imread(i,0)
    img = cv2.resize(image,(28,28))#Because we trained our model with 28*28 pixels training images
    # cv2.imshow("org",image)
    # cv2.imshow("resize",img)
    # cv2.waitKey(0)
    img = img/255  ##Converting the 0 to 255 pixel intensities to 0 to 1 
    img1 = img.reshape((1,28,28)) #because we need too feed to our model image as batch size of 1(because we are feeding 1 image at a time)

    model = load_model(r"C:\Users\oculo\Desktop\Eswar\Projects\number_classification\Number_classification_from_image.h5")
    prediction = model.predict(img1)

    # print(prediction)
    print("Actual image :",name,"\n","Predicted Value",np.argmax(prediction))  #will extracts the index of a max number in the array  and here from 0-9 classes index number can be the predicted number
    # print(classes[int(prediction[0])])
print("Done..!")

###This is model is predicting well when only the image with the background black and number should be in white. same with different colors are not predicting good results

