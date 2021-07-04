#---------------------------------------------------------------------------------------------------
# Project pipeline
# 1. Prepare Model
# 2. Prepare Data Input
# 3. Detect Leaf 
# 4. Recognite Leaf
# 5. Show Result
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Dev by: team RETURN
# Year: 2021
#---------------------------------------------------------------------------------------------------
# Import packages and libraries
#---------------------------------------------------------------------------------------------------
# Library modules
from detection.utils.general import is_colab
import numpy as np
from keras import layers, Model
import keras
import cv2
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
#---------------------------------------------------------------------------------------------------
# User defined modules
from detection import detectLeaf
from recognition import InceptionV3_t
import ModelConfig
import utils_t

#---------------------------------------------------------------------------------------------------
# Global variables
#---------------------------------------------------------------------------------------------------
img_input = []
boundingbox = []

#---------------------------------------------------------------------------------------------------
# 1. Prepare Model
#---------------------------------------------------------------------------------------------------
print("Step 1. Prepare Model")
detect_model = detectLeaf.loadModel_t()
recog_model, labels_dict, species_dict = InceptionV3_t.loadModel_t()
if species_dict is None:
    print("Name of species is store in ./recognition/labels.csv")

#---------------------------------------------------------------------------------------------------
# 2. Prepare Data Input
#---------------------------------------------------------------------------------------------------
print("Step 2. Prepare Data Input")
#img_input = utils_t.load_input(imgpath=ModelConfig.input_path)

#---------------------------------------------------------------------------------------------------
# 3. Detect Leaf 
#---------------------------------------------------------------------------------------------------
print("Step 3. Detect Leaf")
list_boundingbox, img_input = detectLeaf.detect_t(detect_model)
print("Number of Detected Image:", len(img_input))

#---------------------------------------------------------------------------------------------------
# 4. Recognite Leaf
#---------------------------------------------------------------------------------------------------
# 5. Show Result
#---------------------------------------------------------------------------------------------------
print("Step 4 - 5. Recognite Leaf - Show Result")  
idx_input_img = 0 
nleaf = 0
for boundingbox in list_boundingbox: 
    for leafpos in boundingbox:
        nleaf = nleaf + 1
        # Crop leaf area for recognition
        leaf_img = img_input[idx_input_img][leafpos[1]:leafpos[3], leafpos[0]:leafpos[2]]  
        show_img = leaf_img.copy()
        show_img = cv2.resize(show_img, (int(448*leaf_img.shape[1]/leaf_img.shape[0]), 448))

        # Preprocessing image 
        leaf_img = np.array(leaf_img).astype('float32')/255
        leaf_img = transform.resize(leaf_img, (ModelConfig.InceptionConfig_t.img_width, ModelConfig.InceptionConfig_t.img_height, 3))
        leaf_img = np.expand_dims(leaf_img, axis=0)

        # Predict leaf
        leaf_labels_raw = recog_model.predict(leaf_img)

        # Processing leaf labels
        label = list(labels_dict.keys())[list(labels_dict.values()).index(np.argmax(leaf_labels_raw))]
        if species_dict is None:
            name = ""
        else:
            name = list(species_dict.keys())[list(species_dict.values()).index(np.array(label))]
        print("Leaf Recognited: " + str(label) + " - " + str(name))

        # Show leaf information and show image
        cv2.putText(show_img, str(label), (5, 25),cv2.FONT_HERSHEY_SIMPLEX, 2e-3 * show_img.shape[0], (0,0,255), 2)
        cv2.putText(show_img, str(name), (5, 50),cv2.FONT_HERSHEY_SIMPLEX, 2e-3 * show_img.shape[0], (0,0,255), 1) 
        cv2.imwrite(ModelConfig.output_path + str(label) + "_" + name + "_" + str(nleaf) + ".jpg", show_img)
        if ModelConfig.is_colab:
            plt.imshow(show_img), plt.title(str(label) + "_" + name)
            plt.show()
        else:
            cv2.imshow(name, show_img)
            cv2.waitKey(0)  
    idx_input_img = idx_input_img + 1
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------