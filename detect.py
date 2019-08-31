from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import src.demo
from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def nor_coor3d(pred,A,B,C):
  pred = np.array(pred)
  if A == 0 and C == 0:
    m, n, t, x, y, z = (1, 0, 0, 0, 1, 0)
  else:
    m = C / np.sqrt(A ** 2 + C ** 2)
    n = 0
    t = -1 * A / np.sqrt(A ** 2 + C ** 2)
    k = np.sqrt((A**2+C**2)*(A**2+B**2+C**2))
    x = -1 * A * B/ k
    y = (A ** 2 + C ** 2) / k
    z = -1 * B * C / k
  ax = np.array([[m,n,t],[x,y,z]])
  pred3d = np.matmul(pred, ax.T)

  return pred3d

def projection2d(coor,A,B,C):
  p = np.array([[A,B,C]])
  coor = np.array(coor)
  t = np.matmul(coor,p.T)
  k = np.matmul(t,p)
  k = k / sum(sum(p**2))
  return coor - k

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            temp_img = cv2.imread(path)
            circles = []
            for obj_i in range(len(detections)):
                if detections[obj_i, 6] == 0:
                    cropped = temp_img[int(detections[obj_i,1]):int(detections[obj_i,3]),int(detections[obj_i,0]):int(detections[obj_i,2]),:]
                    pred, pred_3d = src.demo.demo_image(cropped)

                    points = pred_3d.reshape(-1,3)
                    x, y, z = np.zeros((3, points.shape[0]))
                    oo = float('inf')
                    xmax, ymax, zmax = -oo, -oo, -oo
                    xmin, ymin, zmin = oo, oo, oo

                    for j in range(points.shape[0]):
                        x[j] = points[j, 0].copy()
                        y[j] = points[j, 2].copy()
                        z[j] = - points[j, 1].copy() + 0.5
                        xmax = max(x[j], xmax)
                        ymax = max(y[j], ymax)
                        zmax = max(z[j], zmax)
                        xmin = min(x[j], xmin)
                        ymin = min(y[j], ymin)
                        zmin = min(z[j], zmin)
                    cena, cenb = ((xmax + xmin) / 2, (ymax + ymin) / 2)
                    r = (np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2) / 2) * 1.2
                    theta = np.arange(0, 2 * np.pi, 0.02)
                    x = cena + r * np.cos(theta)
                    y = cenb + r * np.sin(theta)

                    zl = zmin + 0 * theta
                    zh = (zmax + 0 * theta) * 1.05
                    zl = 0.5 - zl
                    zh = 0.5 - zh
                    circlel = []
                    circleh = []
                    for i in range(len(theta)):
                        circlel.append([x[i], zl[i], y[i]])
                        circleh.append([x[i], zh[i], y[i]])
                    circlel = np.array(circlel)
                    circleh = np.array(circleh)

                    A, B, C = -0.3125, 0.15625, 0.9375
                    circlel = projection2d(circlel, A, B, C)
                    circleh = projection2d(circleh, A, B, C)
                    circlel = nor_coor3d(circlel, A, B, C)
                    circleh = nor_coor3d(circleh, A, B, C)

                    XG = float(detections[obj_i, 0] + pred[6,0])
                    YG = float(detections[obj_i, 1] + pred[6,1])

                    max_x2d = max(pred[:, 0])
                    min_x2d = min(pred[:, 0])
                    max_y2d = max(pred[:, 1])
                    min_y2d = min(pred[:, 1])
                    alpha = max(max_x2d - min_x2d, max_y2d - min_y2d)/(zmax-zmin)

                    circlel = circlel * alpha

                    circlel[:, 0] = circlel[:, 0] + XG
                    circlel[:, 1] = circlel[:, 1] + YG

                    circleh = circleh * alpha
                    circleh[:, 0] = circleh[:, 0] + XG
                    circleh[:, 1] = circleh[:, 1] + YG

                    circlel = (circlel.reshape(circlel.shape[0], -1)).astype(np.int32)
                    circleh = (circleh.reshape(circleh.shape[0], -1)).astype(np.int32)

                    circles.append(circlel)
                    circles.append(circleh)
        for item in range(len(circles)):
            for coor in circles[item]:
                if item % 2 == 0:
                    cv2.circle(img, (coor[0], coor[1]), 2, (0, 0, 255), -1)
                else:
                    cv2.circle(img, (coor[0], coor[1]), 2, (0, 255, 0), -1)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        plt.close()
