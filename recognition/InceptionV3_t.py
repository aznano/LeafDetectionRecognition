import os
import keras
import numpy as np
import shutil
import random
import pandas as pd
# from tensorflow.keras.applications.resnet50 import ResNet50
# import tensorflow as tf
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras import layers, Model
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.callbacks import ModelCheckpoint
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.applications.inception_v3 import InceptionV3

# Other way import packages and libraries
# import tensorflow as tf
from keras import layers, Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
# Package for evaluate
from sklearn.metrics import accuracy_score, classification_report


# Model parameters
import ModelConfig

def InceptionV3Leaf(nb_categories):
    """
    Create InceptionV3 model architecture.
    
    Args:
        nb_categories: Number of categories. Use to create Ouput Layer
    Returns:
        model: Definition of InceptionV3 model
    """
    # Create base model
    base_model = InceptionV3(input_shape = (ModelConfig.InceptionConfig_t.img_height, ModelConfig.InceptionConfig_t.img_width, 3), # Shape of our images
                            include_top = False, # Leave out the last fully connected layer
                            weights = ModelConfig.InceptionConfig_t.pretrain_weight_path)
    for layer in base_model.layers:
        layer.trainable = False

    # Complete our model
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(base_model.output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)                  
    # Add a final softmax layer for classification
    #x = layers.Dense(nb_categories, activation='softmax')(x)  
    x = layers.Dense(nb_categories, activation='sigmoid')(x)

    model = keras.models.Model(base_model.input, x)

    model.compile(optimizer = RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])
    #model.compile(optimizer = RMSprop(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])
    
    model.summary()

    return model
def train_t(model):
    """
    Train model, save dictionary of labels and save model trained weights.

    Args:
        model: Defined InceptionV3 model
    Returns:
        None
    """
    #Number of images to load at each iteration
    # only rescaling
    train_datagen =  ImageDataGenerator(rescale=1./255)
    test_datagen =  ImageDataGenerator(rescale=1./255)
    # these are generators for train/test data that will read pictures found in the defined subfolders of 'data/'
    print('Total number of images for "training":')
    train_generator = train_datagen.flow_from_directory(ModelConfig.InceptionConfig_t.train_data_dir,
                                                        target_size = (ModelConfig.InceptionConfig_t.img_height, ModelConfig.InceptionConfig_t.img_width),
                                                        batch_size = ModelConfig.InceptionConfig_t.batch_size_t, 
                                                        class_mode = "categorical")
    print('Total number of images for "validation":')
    val_generator = test_datagen.flow_from_directory(ModelConfig.InceptionConfig_t.val_data_dir,
                                                    target_size = (ModelConfig.InceptionConfig_t.img_height, ModelConfig.InceptionConfig_t.img_width),
                                                    batch_size = ModelConfig.InceptionConfig_t.batch_size_t,
                                                    class_mode = "categorical",
                                                    shuffle=False)
    # Get and save labels dictionary
    label_dict = train_generator.class_indices
    df = pd.DataFrame.from_dict(label_dict, orient="index")
    df.to_csv(ModelConfig.InceptionConfig_t.label_dict_csv)
    #--------------------------------------------------
    # Train model
    inc_history = model.fit_generator(train_generator,
                                    validation_data = val_generator, 
                                    epochs = 10)
    # Save trained model: it will save following:

    # 1. The architecture of the model, allowing to create the model.
    # 2. The weights of the model.
    # 3. The training configuration of the model (loss, optimizer).
    # 4. The state of the optimizer, allowing training to resume from where you left before.
    model.save(ModelConfig.InceptionConfig_t.weight_path)

def loadModel_t():
    """
    Use to load InceptionV3 model weights.

    Args:
        None
    Returns:
        recog_model: InceptionV3 model
        labels_dict: Dictionary of labels
        species_dict: Dictionary of species's name
    """
    # Create Model architechture
    recog_model = InceptionV3Leaf(ModelConfig.InceptionConfig_t.nb_categories)
    
    # Load the model weights
    recog_model = keras.models.load_model(ModelConfig.InceptionConfig_t.weight_path)

    # Read labels 
    labels_dict_df = pd.read_csv(ModelConfig.InceptionConfig_t.label_dict_csv, index_col=0)
    labels_dict = labels_dict_df.to_dict("split")
    labels_dict = dict(zip(labels_dict["index"], labels_dict["data"]))

    # Read species name
    if os.path.isfile(ModelConfig.InceptionConfig_t.species_dict_csv):
        species_dict_df = pd.read_csv(ModelConfig.InceptionConfig_t.species_dict_csv, index_col=0)
        species_dict = species_dict_df.to_dict("split")
        species_dict = dict(zip(species_dict["index"], species_dict["data"]))
    else:
        species_dict = None
    return recog_model, labels_dict, species_dict
def evaluate_t(model, visual=False):
    """
    Evaluate accuracy of model
    Args:
        model: ready InceptionV3 model
        visual: True or False. Visualize test dataset or not
    Returns:
        None
    """
    test_datagen =  ImageDataGenerator(rescale=1./255)
    print('Total number of images for "testing":')
    test_generator = test_datagen.flow_from_directory(ModelConfig.InceptionConfig_t.test_data_dir,
                                                    target_size = (ModelConfig.InceptionConfig_t.img_height, ModelConfig.InceptionConfig_t.img_width),
                                                    batch_size = ModelConfig.InceptionConfig_t.batch_size_t,
                                                    class_mode = "categorical",
                                                    shuffle=False)
    # Number of image using for testing
    numtest = test_generator.samples
    # Prediction
    Y_pred = model.predict_generator(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_test = test_generator.classes
    print("Default index of label:", y_test)
    print("Predicted index of label:", y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("InceptionV3 - Accuracy in test set: %0.3f%% " % (accuracy * 100))
    
    #---------------------
    if visual:
        idx_img = 0
        true_label = []
        idx_label = []
        print("Visualizing...")
        # Generate plots for samples
        for sample, numclass  in test_generator:
            idx_label = np.argmax(numclass,axis=1)
            print("sample.shape=",sample.shape)
            print("corresponding index of label=",idx_label)
            i = 0
            for img in sample:
            # Generate a plot
                plt.imshow(img), plt.title("true_label: "+str(idx_label[i])+"|predict: "+str(y_pred[idx_img]))
                plt.show()
                idx_img = idx_img + 1
                i = i + 1
                if idx_img >= numtest:
                    break
            if idx_img >= numtest:
                break
