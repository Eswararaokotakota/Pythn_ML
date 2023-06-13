import cv2

from tensorflow.keras.models import load_model
from tensorflow import keras

path = r"C:\Users\oculo\Desktop\Eswar\Projects\Python_ML_Projects\dogs_cats_classification\data_for_testing\catttt.jpg"
image = cv2.imread(path)
image = cv2.resize(image,(256,256))  #because we trained our model with thish shape

##Because we trained the model with the batches we need to reshape the single image to make it look like one image in the batch
image_input = image.reshape((1,256,256,3))

history = load_model(r"C:\Users\oculo\Desktop\Eswar\large_datasets&models\models\Dogs_Cats_Classification_Model.h5")
prediction = history.predict(image_input)

classes = ["Cat","Dog"]
print("----------------------",classes[int(prediction)],"----------------------",)
print(history.summary())

####### Model is working well ###########

######Let's diaplay the model structure flow chart

from keras.utils.vis_utils import plot_model
plot_model(history, to_file='model.png', show_shapes=True)   ##This will save a model.png file that consists the flow chart image of the model 


# import matplotlib.pyplot as plt
# plt.plot(history.history["accuracy"],color="red",label="Train")
# plt.plot(history.history["val_accuracy"],color="blue",label="Test")
# plt.legend()
# plt.show()