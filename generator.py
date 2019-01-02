from scipy.ndimage.filters import gaussian_filter
from scipy import spatial
import numpy as np
import scipy
import h5py
import os
import cv2
import glob


def gaussian_filter_density(gt):
    # Generates a density map using Gaussian filter transformation
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    # FInd out the K nearest neighbours using a KDTree
    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048

    # build kdtree
    tree = spatial.KDTree(pts.copy(), leafsize=leafsize)

    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point

        # Convolve with the gaussian filter
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density


def path_generator(root):
    part_A_train = os.path.join(root, 'part_A_final','train_data', 'images')
    part_A_test = os.path.join(root, 'part_A_final','test_data', 'images')
    part_B_train = os.path.join(root, 'part_B_final','train_data', 'images')
    part_B_test = os.path.join(root, 'part_B_final','test_data', 'images')
    
    path_sets = [part_A_train,part_A_test]

    img_paths = []

    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(str(img_path))

    print("Total images : ", len(img_paths))
    return img_paths


def get_input(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = img / 255.0
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

    return img


def get_gt(path):
    gt = h5py.File(path, 'r')
    gt = np.asarray(gt['density'])
    gt = cv2.resize(gt,(28,28))*64
    gt = np.expand_dims(gt, axis=3)
    return gt


# Image data generator
def image_generator(paths, batch_size=64):
    while True:
        input_paths = np.random.choice(a=paths, size=batch_size)
        batch_input = []
        batch_output = []

        for input_path in input_paths:

            img_data = get_input(input_path)
            gt_data = get_gt(input_path.replace('.jpg', '.h5').replace('images', 'ground_truth'))
            batch_input += [img_data]
            batch_output += [gt_data]

            batch_input= np.array(batch_input)
            batch_output = np.array(batch_output)

        yield (batch_input, batch_output)


def save_mod(model, str1, str2):
    model.save_weights(str1)
    model_json = model.to_json()

    with open(str2, "w") as json_file:
        json_file.write(model_json)



# def preprocess_input(image, target):
#     # crop image and ground-truth
#     #crop_size = (int(image.shape[0] / 2), int(image.shape[1] / 2))
#     crop_size=(224,224)
#     if random.randint(0, 9) <= -1:
#         dx = int(random.randint(0, 1) * image.shape[0] * 1. / 2)
#         dy = int(random.randint(0, 1) * image.shape[1] * 1. / 2)
#     else:
#         dx = int(random.random() * image.shape[0] * 1. / 2)
#         dy = int(random.random() * image.shape[1] * 1. / 2)
#
#     # print(crop_size , dx , dy)
#     img = image[dx: crop_size[0] + dx, dy:crop_size[1] + dy]
#
#     target_aug = target[dx:crop_size[0] + dx, dy:crop_size[1] + dy]
#     # print(img.shape)
#
#     return (img, target_aug)