# Import packages and libraries
import os
import cv2
from PIL import Image
import numpy as np
from skimage import transform
import ModelConfig


def load(filename):
    """
    Read Single Image in Keras model input Format.

    Args:
        filename: path of image
    Returns:
        np_image: output image
    """
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (ModelConfig.InceptionConfig_t.img_width, 
                                            ModelConfig.InceptionConfig_t.img_height, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def load_input(imgpath=""):
    """
    Read Multiple Image from a folder.
    
    Args:
        imgpath: path to input folder
    Returns:
        img_list: list of readed image
    """
    img_list = []
    for file in os.listdir(imgpath):
        filepath = imgpath + "/" + file
        print("filepath=", filepath)
        img = load(filepath)
        img_list.append(img)
    return img_list
