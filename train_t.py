from detection import yolotrain_t
from recognition import inceptiontrain_t
import argparse

def main(opt):
    if not opt.yolo and not opt.inception:
        print("Training Yolo Detection...")
        yolotrain_t.train_yolo()
        print("Training Inception Recognition...")
        inceptiontrain_t.train_inception()
        print("Complete training process!")
    elif opt.yolo:
        print("Training Only Yolo Detection...")
        yolotrain_t.train_yolo()
        print("Complete training process!")
    elif opt.inception:
        print("Training Inception Recognition...")
        inceptiontrain_t.train_inception()
        print("Complete training process!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', action='store_true', help='only train yolo')
    parser.add_argument('--inception', action='store_true', help='only train inception')
    opt = parser.parse_args()
    
    main(opt)