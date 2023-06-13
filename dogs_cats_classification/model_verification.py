import cv2

from tensorflow.keras.models import load_model

path = r"C:\Users\oculo\Desktop\Eswar\Projects\dogs_cats_classification\data_for_testing\dog_as_cat.jfif"
image = cv2.imread(path)
image = cv2.resize(image,(256,256))  #because we trained our model with thish shape

##Because we trained the model with the batches we need to reshape the single image to make it look like one image in the batch
image_input = image.reshape((1,256,256,3))

history = load_model(r"C:\Users\oculo\Desktop\Eswar\Projects\dogs_cats_classification\Dogs_Cats_Classification_Model.h5")
prediction = history.predict(image_input)

classes = ["Cat","Dog"]
print("----------------------",classes[int(prediction)],"----------------------",)
print(history.summary())

####### Model is working well ###########

# import matplotlib.pyplot as plt
# plt.plot(history.history["accuracy"],color="red",label="Train")
# plt.plot(history.history["val_accuracy"],color="blue",label="Test")
# plt.legend()
# plt.show()