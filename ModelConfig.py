from logging import FATAL, fatal
import os
import torch

input_path = "./input/"
output_path = "./output/"
is_colab = True # Change to False if run on physical computer

class InceptionConfig_t:
    # Dataset
    train_data_dir = "recognition/dataset/resnet/train/"
    val_data_dir = "recognition/dataset/resnet/val/"
    test_data_dir = "recognition/dataset/resnet/test/"
    category_names = os.listdir('recognition/dataset/resnet/train/')
    nb_categories = len(category_names)

    # Labels 
    label_dict_csv = "recognition/labels.csv"
    species_dict_csv = "recognition/species.csv"
    
    # Model parameters
    img_height, img_width = 224,224
    batch_size_t = 32
    weight_path = "recognition/train/weights/model_inceptionv3.h5"
    pretrain_weight_path = "recognition/train/weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Set model parameters
#'cuda' if torch.cuda.is_available() else 'cpu'
class YoloConfig_t:
    weights = 'detection/train/weights/best.pt'
    source = './input/'    # Input image path( to detect)
    img_size = 416
    conf_thres = 0.5
    iou_thres = 0.5
    device = ''
    view_img = False
    save_txt = False
    save_conf = False
    classes = None
    agnostic_nms = False
    augment = False
    update = True
    project = 'detection/runs'   # Detection Result path
    name = 'exp'
    exist_ok = False
class YoloTrainConfig_t:
    # Training process
    adam=False
    batch_size=16
    bucket=''
    cache_images=True
    cfg=''
    data='detection/dataset/yolov5/data.yaml'
    device=''
    entity=None
    epochs=100
    evolve=False
    exist_ok=False
    global_rank=-1
    hyp='detection/data/hyps/hyp.scratch.yaml'
    image_weights=False
    img_size=[416, 416]
    linear_lr=False
    local_rank=-1
    log_artifacts=False
    log_imgs=16
    multi_scale=False
    name=''
    noautoanchor=False
    nosave=False
    notest=False
    project='detection/train/'
    quad=False
    rect=False
    resume=False
    save_dir='detection/train/'
    single_cls=False
    sync_bn=False
    total_batch_size=16
    weights='yolov5l.pt'
    workers=8
    world_size=1
