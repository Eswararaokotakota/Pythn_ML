import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
import sys

import numpy as np 
import matplotlib.pyplot as plt
import cv2

(X_train,y_train),(X_test,y_test)= keras.datasets.mnist.load_data()

X_train = X_train/255
X_test = X_test/255



# print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
# plt.imshow(X_train[0])
# print(y_train[0])
# plt.show()
img = X_test[0]
print(img.shape)
cv2.imshow("asbd",img)
cv2.waitKey(0)
cv2.imwrite("i.png",img)

sys.exit("stopped")  #program execution will stops here (for testin purpose)

model = Sequential()

################# Architecture of our cnn ##################
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))  #activation softmax used because there is morethan two outcomes from the prediction(ie. 0 to 9)
################# Architecture of our cnn end ##################
model.summary()

### Now let's start compailing our cnn ###
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])  ##sparse_categorical_crossentropy is specifically designed for multiclass classificaton

##Fitting the model
history = model.fit(X_train,y_train,batch_size=64,epochs=10,validation_data=(X_test,y_test))  ###Batch size will tells that how meny samples should process at one time (increasing this will need more memory ppower)
##And epochs will defines that how meny times the training process should done. increasing this will cause more time for traning. keep on eye on acccuracy after increasing epoochs.
##if it is increased(epochs) tooo much the accuracy of test data will start decreasing because of overfitting
###NOTE: By changing these two (batch_size and epochs) values we can increase the accuracy (But need to perform tral and error opertions)

test_loss,test_accuracy = model.evaluate(X_test,y_test,verbose=1)  #In this we need to provide the test data which is not given to the model during training process so this is the new data for ti model
##Here the model.evaluate takes the test data and test labes and performs the testing operation with the model and gives us the loss and accuracy of the model to make us to understand the performance of our model
##And verbose=1 provides the information of loss and accuracy in the terminal, if it is set to 0. then it shows nothing during the evaluation, if it set to 2 it will provides small summary aabout the evaluation results.
##For better model observations it is better to use verbose=1
print("Test loss:",test_loss,"\n","Test accuracy:",test_accuracy)

##Let's plot the accuracy of traning and testig data for comparision to understand how good is our model perfrming in test data(which is trained by train data)
train_accuracy = history.history["accuracy"]
test_accuracy = history.history["val_accuracy"]
plt.plot(train_accuracy,label="Train accuracy")
plt.plot(test_accuracy,label="Test accuracy")
plt.legend()
plt.show()
####same as for loss validation(checking between tran data and test data)
train_loss = history.history["loss"]
test_loss = history.history["val_loss"]
plt.plot(train_loss,label="Train loss")
plt.plot(test_loss,label="Test loss")
plt.legend()
plt.show()

###ANDDDDDDDDDD let's save our model for future prediction by just giving input image it shoud tell what number it is (only sngle digit numbers works i think)
model.save("Number_classification_from_image.h5")


print("Done..!")
