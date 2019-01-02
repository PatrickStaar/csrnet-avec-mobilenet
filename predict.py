# -*- coding: utf-8 -*-
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
from keras.models import model_from_json
from generator import get_input,get_gt

def load_model():
    # Function to load and return neural network model 
    json_file = open('models/Model_mobile.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights/premier.finetuned.1000iter.h5")
    return loaded_model

model = load_model()
# cv2.namedWindow('Video')
# cap = cv2.VideoCapture('test_images/sample.mp4')
# count=0
# while cap.isOpened():
#     fine,frame = cap.read()
#     if fine:
#         frame=create_img(frame)
#         hmap = model.predict(frame)
#         print(np.sum(hmap)*256)
#         hmap = hmap.reshape(hmap.shape[1],hmap.shape[2])
#         hmap= (hmap*255).astype(np.uint8)
#         hmap=cv2.resize(hmap,(640,360))
#
#         cv2.imshow('Video',hmap*10)
#         cv2.waitKey(30)
#     else:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

im_path='./data/part_A_final/train_data/images/IMG_3.jpg'
gt_path='./data/part_A_final/train_data/ground_truth/IMG_3.h5'

im = get_input(im_path)
im = np.expand_dims(im,axis  = 0)

# groundtruth = h5py.File(gt_path, 'r')
# groundtruth = np.asarray(groundtruth['density'])
groundtruth = get_gt(gt_path)

ori_im=cv2.imread(im_path)
ori_shape=(ori_im.shape[0],ori_im.shape[1])

output = model.predict(im)
print(output.shape)
output=output.reshape(output.shape[1],output.shape[2])

groundtruth = groundtruth.reshape(groundtruth.shape[0],groundtruth.shape[1])
#output = cv2.resize(output,(ori_shape[1],ori_shape[0]))

prediction=np.sum(output)
actual=np.sum(groundtruth)

plt.figure(1)
gt=plt.subplot(1,3,1).imshow(groundtruth,cmap=c.jet)
pre=plt.subplot(1,3,2).imshow(output,cmap=c.jet)
img=plt.subplot(1,3,3).imshow(ori_im)
plt.show()

print('original size:',ori_shape)
print('predicted num:',prediction)
print('actual number:',actual)
