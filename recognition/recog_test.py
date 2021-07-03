#--------------------------------------------------------------------
# Import packages and libraries
import numpy as np
from keras import layers, Model, Input
import keras
import cv2

from recognition import InceptionV3_t
import ModelConfig
import utils_t

#--------------------------------------------------------------------
# Global variables
img_input = []
boundingbox = []



#--------------------------------------------------------------------
# 1. Prepare Model
print("Step 1. Prepare Model")
recog_model, labels_dict, species_dict = InceptionV3_t.loadModel_t()
# Replace input layer of model
# recog_model.layers.pop(0)
# newInput = keras.Input(batch_shape=(None,ModelConfig.InceptionConfig_t.img_height,ModelConfig.InceptionConfig_t.img_width,3))
# newOutputs = recog_model(newInput)
# newRecogModel = Model(newInput, newOutputs)


#--------------------------------------------------------------------
# 2. Prepare Data Input
print("Step 2. Prepare Data Input")
img_input = utils_t.load_input(imgpath=ModelConfig.input_path)

#--------------------------------------------------------------------
# 3. Recognite Leaf
n_img = len(img_input)
print("Number of image: ", n_img)
print("Step 4. Recogniting Leaf")   
for img in img_input:
    leaf_labels_raw = recog_model.predict(img)
    leaf_labels_raw = np.argmax(leaf_labels_raw)
    label = list(labels_dict.keys())[list(labels_dict.values()).index(np.argmax(leaf_labels_raw))]
    name = list(species_dict.keys())[list(species_dict.values()).index(np.array(label))]
    print("Leaf Recognited: " + str(label) + " - " + str(name))


