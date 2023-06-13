
 ##Firt we willl import the required modules to perform operation
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

#now we need to provide the image path for this we can easily provide the path and do a forloop and access images one by one
## but the problem here is it will take more time and computational power to overcome this problem a function called "Generators are introduced"
## using the generators function the data will be devided into batches and then will be fed to the model (lern more about generators in cnn if you have any doubt)

#Grnerators
#Assigning the paths of datasets for train and test
Train_dataset= keras.utils.image_dataset_from_directory(
    directory=r"C:\Users\oculo\Desktop\Eswar\Projects\dogs_cats_classification\kaggle_dogs_cats_data\train", ##path of train data
    labels="inferred",
    label_mode="int",   ## this will assaign 0 to cats and 1 to cats viseversa
    batch_size=32,
    image_size=(256,256)  ##Here we are accepting the every image in this size. if not there will be problem in the future detection and training
)
#same for test
Test_dataset = keras.utils.image_dataset_from_directory(
     directory=r"C:\Users\oculo\Desktop\Eswar\Projects\dogs_cats_classification\kaggle_dogs_cats_data\test",
    labels="inferred",
    label_mode="int",   
    batch_size=32,
    image_size=(256,256)
)
##In the above process we assigned the train test data folders to the variables


##Now In the cnn the pixel values should be in between 0 to 1 But our image pixels will be varies from 0 to 255 so we are converting those 0-255 values to 0-1 respectively
def conversion(image,label):
    image=tf.cast(image/255. , tf.float32)
    return image,label
Train_dataset = Train_dataset.map(conversion)
Test_dataset = Test_dataset.map(conversion)


##Now we will create a CNN model
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding="valid",activation="relu",input_shape=(256,256,3)))
model.add(BatchNormalization())  ##Additionally added to improve the training performance for better accuracy
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding="valid"))

model.add(Conv2D(64,kernel_size=(3,3),padding="valid",activation="relu"))
model.add(BatchNormalization())##
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding="valid"))

model.add(Conv2D(128,kernel_size=(3,3),padding="valid",activation="relu"))
model.add(BatchNormalization())##
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding="valid"))
#We repeated the convolution and maxpooling 3times by giving the filters for 1st step 32,2nd step 64, 128

model.add(Flatten()) #we are converting the 2d image to 1d now

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.1))##Additionally added to improve the training performance for better accuracy
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.1))##
model.add(Dense(1,activation="sigmoid"))# This is the final nuron which gives us the prediction output

#we have successfully created our first cnn model here ....................

# model.summary()  #This will gives us the summary of this cnn model process how meny filters are applied and how meny samples are generated so onn..

##Lets compail the model
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

##Now it's time to fit our first CNN model
history = model.fit(Train_dataset,epochs=30,validation_data=Test_dataset) #it will take some time to fit the model depends of cpu or gpu capacity
#epochs will tels the model that how meny times the model should iterate through tha data  (increasing this walue will increases the accuracy a little)
   

##After sometime our cat dog classification cnn model is ready....


#### To save the model you can use the below command
model.save("Dogs_Cats_Classification_Model.h5")


##To get the every detail of model just use 
model.summary()##used again


##Let's visualize the accuracy on training and testing data
# import matplotlib.pyplot as plt
# plt.plot(dc_classifier_model.dc_classifier_model["accuracy"],color="red",label="Train")
# plt.plot(dc_classifier_model.dc_classifier_model["val_accuracy"],color="blue",label="Test")
# plt.legend()
# plt.show()
