import os
import warnings
warnings.filterwarnings('ignore')
from os.path import join
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keras.backend as K
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from samples.dic import dic
from samples.custo import custo
# %matplotlib inline 

def miou(y_true, y_pred):
    label = tf.reshape(y_true, y_true.shape)
    pred = tf.reshape(y_pred, y_pred.shape)

    iou_op = tf.metrics.mean_iou(label, pred ,2)
    with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer()) 
        K.get_session().run(tf.local_variables_initializer())

        sess.run(iou_op)
        mean_iou, conf_mat = sess.run(iou_op)
        return mean_iou

def clip2one(masks):
    mask = masks.sum(axis=-1)
    return np.where(mask==0, 0, 1)

def load_data(dataset_name):
    if dataset_name in ['dsb', 'fluo_n2dh']:
        data_dir = '/root/data/{}'.format(dataset_name)
        dataset = custo.CustoDataset()
        dataset.load_custo(data_dir, 'valid')
    elif dataset_name=='dic':
        data_dir = '/root/data/{}'.format(dataset_name)
        dataset = dic.DicDataset()
        dataset.load_dic(data_dir, 'val')
    dataset.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    return dataset

DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
config = dic.DicConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

dataset_names = ['dsb', 'fluo_n2dh', 'dic']
Miou, CA = [], []
for dataset_name in dataset_names:
    MODEL_DIR = "/root/GitRepo/Mask_RCNN-master/logs/{}".format(dataset_name)
    WEIGHTS_PATH = join(MODEL_DIR, 'mask_rcnn_{}.h5'.format(dataset_name))  # TODO: update this path
    # DIC_DIR = os.path.join(ROOT_DIR, "datasets/{}")

    dataset = load_data(dataset_name)

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(WEIGHTS_PATH, by_name=True)

    # Load and display random samples
    miu, mca = [], []
    for iid, image_id in enumrate(dataset.image_ids):
        if iid>=50:
            break
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        # visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
        # print(image.shape, mask.shape)
        pred = model.detect([image], verbose=1)[0]['masks']
        t, p = clip2one(mask), clip2one(pred)
        miu.append(miou(t, p))

        n_cells, p_cells = len(class_ids), pred.shape[-1]
        mca.append(abs(n_cells-p_cells))
    Miou.append(np.mean(miu))
    CA.append(np.mean(mca))

print(dataset_names)
print(Miou)
print(CA)


