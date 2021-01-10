from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

class ImgConfig(Config):
    NAME = "images"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 80+1

def seg (img_path):
    MODEL_DIR = 'Mask_RCNN'
    rcnn = MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=ImgConfig())
    rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    img = load_img(img_path)
    img_ = img_to_array(img)
    # make prediction
    results = rcnn.detect([img_], verbose=0)
    # get dictionary for first prediction
    r = results[0]
    bool_ = r['class_ids']==1
    masks = r['masks'].astype(int)[:,:,bool_].astype(bool)
    rois = r['rois'][bool_]
    scores = r['scores'][bool_]
    #display_instances(img_, rois, masks ,r['class_ids'][bool_], class_names,scores)
    return img,masks,rois,scores

#cut white area
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def cut (img,masks):
    img = np.array(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masked_a = np.ma.masked_array(img,np.repeat(masks, img.shape[2]).reshape(img.shape),fill_value=255)
    img_copy = img.copy()
    img_copy[~masked_a.mask]=255
    img = Image.fromarray(img_copy, 'RGB')
    res = trim(img)
#     im = np.array(res)
#     plt.figure(figsize=(15,9))
#     plt.imshow(im)
#     plt.show()
    return res