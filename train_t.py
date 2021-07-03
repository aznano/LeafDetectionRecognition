from detection import yolotrain_t
from recognition import inceptiontrain_t
import ModelConfig

if not ModelConfig.only_train_yolo and not ModelConfig.only_train_inception:
    print("Training Yolo Detection...")
    yolotrain_t.train_yolo()
    print("Training Inception Recognition...")
    inceptiontrain_t.train_inception()
    print("Complete training process!")
elif ModelConfig.only_train_yolo:
    print("Training Only Yolo Detection...")
    yolotrain_t.train_yolo()
    print("Complete training process!")
elif ModelConfig.only_train_inception:
    print("Training Inception Recognition...")
    inceptiontrain_t.train_inception()
    print("Complete training process!")

