import cv2
import numpy as np
from ..utils import image as img_util
def preprocess_image(img_path, img_size=256,annotation= None):
    img = cv2.imread(img_path)/255
    if(annotation!=None):
        img[:, :, 0] = np.multiply(img[:, :, 0], annotation["seg"])
        img[:, :, 1] = np.multiply(img[:, :, 1], annotation["seg"])
        img[:, :, 2] = np.multiply(img[:, :, 2], annotation["seg"])
    # img=img/255
    # Scale the max image size to be img_size
    scale_factor = float(img_size) / np.max(img.shape[:2])
    # print(scale_factor)
    # exit()
    img, _ = img_util.resize_img(img, scale_factor)
    # print(img.shape)
    # exit()
    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    # print(center)

    center = center[::-1]
    # print(center)
    # exit()
    bbox = np.hstack([center - img_size / 2., center + img_size / 2. - 1])
    # print(bbox)
    # exit()
    img = img_util.crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))
    return img
def read_image_and_preprocess(filename,annotation=None):
    img_size=256
    if(annotation==None):
        image_ref_np = preprocess_image(filename)
    else:
        image_ref_np = preprocess_image(filename,img_size,annotation)
    image_ref_np = np.transpose(image_ref_np, (1, 2, 0))
    image_ref_np = image_ref_np[:, :, ::-1]
    return image_ref_np
#--------------------------------------------------------------ugly code need be clean
# def preprocess_image_with_annotation(img_path,annotation,img_size=256):
#     img=cv2.imread(img_path)/255.
#     img[:,:,0]=np.multiply(img[:, :, 0], annotation["seg"])
#     img[:,:,1]=np.multiply(img[:, :, 1], annotation["seg"])
#     img[:,:,2]=np.multiply(img[:, :, 2], annotation["seg"])
#     # img = img / 255.
#
#     # Scale the max image size to be img_size
#     scale_factor = float(img_size) / np.max(img.shape[:2])
#     img, _ = img_util.resize_img(img, scale_factor)
#
#     # Crop img_size x img_size from the center
#     center = np.round(np.array(img.shape[:2]) / 2).astype(int)
#     # img center in (x, y)
#     center = center[::-1]
#     bbox = np.hstack([center - img_size / 2., center + img_size / 2. - 1])
#
#     img = img_util.crop(img, bbox, bgval=1.)
#
#     # Transpose the image to 3xHxW
#     img = np.transpose(img, (2, 0, 1))
#
#     return img
# def read_image_and_preprocess_but_annotation(filename_ref,annotation):
#     image_ref_np = preprocess_image_with_annotation(filename_ref,annotation)
#     image_ref_np = np.transpose(image_ref_np, (1, 2, 0))
#     image_ref_np = image_ref_np[:, :, ::-1]
#     return image_ref_np
def transpose_img(img_path):
    img = cv2.imread(img_path)/255
    img = np.transpose(img, (2, 0, 1))
    return img
