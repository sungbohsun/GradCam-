import cv2
import os
import numpy as np
from tool import cut,seg
from glob import glob
from multiprocessing import Pool
from os import getpid
from tqdm import tqdm
from glob import glob
from PIL import Image
from keras.preprocessing.image import load_img

def f(n,file):
    #try:
    socre = file.split('/')[2].split('_')[0]
    #print("I'm process", getpid())
    k = 0
    img,masks,rois,scores = seg(file)
    for c in range(masks.shape[2]):
        area = rois[c]
        diagonal = ((area[0]-area[2])**2+(area[1]-area[3])**2)**1/2
        if diagonal>90000 and scores[c]>0.97:
            im = cut(img,masks[:,:,c])
            im.save('beauty_seg/{}_{}_{}.jpg'.format(n,k,socre))
    #             print('{}_{}.jpg'.format(n,k))
    #             print('scores =',scores[c])
    #             print('diagonal =',diagonal)
            k += 1
    #         else :
    #             print('segment_jpg/{}_{}.jpg'.format(n,k),'is unqualified ')
#     except:
#         print('error in',n)
#         pass

def padding(im_pth):
    desired_size = 224
    im = np.array(load_img(im_pth))
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    output = Image.fromarray(new_im, 'RGB')
    return output


files = glob('ppt_crawer/Beauty_PttImg_*/*/*.jpg')
n = list(range(len(files)))
file = [c for c in files]
if __name__ == '__main__':  
    
    if not os.path.isdir('beauty_seg'):
        os.mkdir('beauty_seg')
    with Pool(20) as pool:  
        result = pool.starmap(f,zip(n,file))
        
    if not os.path.isdir('beauty_re'):
        os.mkdir('beauty_seg')        
    files = glob('beauty_seg/*')
    for file in tqdm(files):
        re = padding(file)
        re.save('beauty_re/'+file.split('/')[1:][0])