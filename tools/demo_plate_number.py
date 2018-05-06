#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import operator
import math

CLASSES = ('__background__',
           'plate')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'zf_faster_rcnn_iter_10000.caffemodel')}

MIN_CONTOUR_AREA = 100
MIN_CONTOUR_WIDTH = 10
MIN_CONTOUR_HEIGHT = 25

MAX_CONTOUR_WIDTH = 60
MAX_CONTOUR_HEIGHT = 60

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

MIN_ASPECT_RATIO = 0.1
MAX_ASPECT_RATIO = 1.5

class ContourWithData():
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour
    aspectRatio = 0.0
    
    def calculateRectTopLeftPointAndWidthAndHeight(self):     # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight
        self.aspectRatio = float(intWidth) / float(intHeight)

    def checkIfContourIsValid(self):   # this is oversimplified, for a production grade program
        #if self.intRectWidth < MIN_CONTOUR_WIDTH or self.intRectHeight < MIN_CONTOUR_HEIGHT or self.intRectWidth > MAX_CONTOUR_WIDTH or self.intRectHeight > MAX_CONTOUR_HEIGHT: return False  
        if self.fltArea < MIN_CONTOUR_AREA or self.intRectWidth > MAX_CONTOUR_WIDTH or self.intRectHeight < MIN_CONTOUR_HEIGHT: return False        
        #if self.aspectRatio > MIN_ASPECT_RATIO and self.aspectRatio < MAX_ASPECT_RATIO and self.intRectWidth > MIN_CONTOUR_WIDTH and self.intRectWidth < MAX_CONTOUR_WIDTH and self.intRectHeight > MIN_CONTOUR_HEIGHT and self.intRectHeight < MAX_CONTOUR_HEIGHT: return True	
	# much better validity checking would be necessary
        return True






def img_vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        
   # ax.set_title(('{} detections with '
   #               'p({} | box) >= {:.1f}').format(class_name, class_name,
   #                                              thresh),
   #              fontsize=14)
    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    print image_name
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    
    #imgGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)
 
    #kernel = np.ones((1,30), np.uint8)
    #kernel2 = np.ones((2,2), np.uint8)

    #imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    #imgThreshCopy = imgThresh.copy()
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
	
	inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    	if len(inds) == 0:
        	continue


        #fig, ax = plt.subplots(figsize=(12, 12))
        #ax.imshow(im, aspect='equal')
        for i in inds:
            bbox = dets[i, :4]
	    #print 'bbox',bbox
            score = dets[i, -1]
	    cropped = im[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            text = detect_char(cropped)

	    #cv2.imshow('cropped',cropped)
            ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(text),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        
   # ax.set_title(('{} detections with '
   #               'p({} | box) >= {:.1f}').format(class_name, class_name,
   #                                              thresh),
   #              fontsize=14)
   
        plt.axis('off')
        plt.tight_layout()
        plt.draw()

	

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args

def maxContrast(imgGray):
    height, width = imgGray.shape
    
    TopHatKernel = np.zeros((height, width, 1), np.uint8)
    BlackHatKernel = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGray, TopHatKernel)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, BlackHatKernel)

    #cv2.imshow("Top-Minus", imgGrayscalePlusTopHatMinusBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)    

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    #print imgValue,'\n'
    #imgValue += 50
    return imgValue

def detect_char(cropped):
    allContoursWithData = []
    validContoursWithData = []
  
    try:
       npaCls = np.loadtxt("/home/uofsko-lab/Downloads/py-faster-rcnn-master/tools/classifications.txt", np.float32)
    except:
       print "Unable to open classification.txt.\n"
       return

    try:
       npaFlat = np.loadtxt("/home/uofsko-lab/Downloads/py-faster-rcnn-master/tools/flattened_images.txt", np.float32)
    except:
       print "Unable to open flattened_images.txt.\n"
       return

    npaCls = npaCls.reshape((npaCls.size, 1))

    kNN = cv2.ml.KNearest_create()

    kNN.train(npaFlat, cv2.ml.ROW_SAMPLE, npaCls)

    TARGET_PIXEL_AREA = 24000.0

    ratio = float(cropped.shape[1]) / float(cropped.shape[0])
    new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
    new_w = int((new_h * ratio) + 0.5)

    imgTesting = cv2.resize(cropped, (new_w,new_h))
    
    imgTestCopy = imgTesting.copy()
    #cv2.imshow("imgTesting",imgTesting)
    imgGray = extractValue(imgTesting)
    imgGrayMax = maxContrast(imgGray)
    imgBlur = cv2.GaussianBlur(imgGrayMax, (5,5), 0)
 
    kernel = np.ones((130,1), np.uint8)
    kernel2 = np.ones((1,70), np.uint8)
    kernel3 = np.ones((2,3), np.uint8)

    #imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    thresholdValue, imgThresh = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresholdValue, imgThresh = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #cv2.imshow("original_Thresh", imgThresh)
    imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_TOPHAT, kernel)
    imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_TOPHAT, kernel2)
    
    #imgThresh = cv2.bitwise_not(imgThresh)
    #imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel3)
    #imgThresh = cv2.bitwise_not(imgThresh)
    imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel3)
    
    cv2.imshow("Thresh", imgThresh)
    #cv2.imwrite(os.path.join('/home/uofsko-lab/Documents','Thresh.jpg'),imgThresh)
    imgThreshCopy = imgThresh.copy() 
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for npaContour in npaContours:
        contourWithData = ContourWithData()
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourIsValid():
           validContoursWithData.append(contourWithData)

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))
    strFinalString = ""

    for contourWithData in validContoursWithData:
        cv2.rectangle(imgTestCopy, (contourWithData.intRectX, contourWithData.intRectY), (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight), (0, 255, 0), 2)
        
        charCrop = imgTesting[int(contourWithData.intRectY):int(contourWithData.intRectY + contourWithData.intRectHeight),int(contourWithData.intRectX):int(contourWithData.intRectX + contourWithData.intRectWidth)]
       
        #print contourWithData.intRectWidth
        #print contourWithData.intRectHeight

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
        
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
 
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNN.findNearest(npaROIResized, k = 1)

        strCurrentChar = str(chr(int(npaResults[0][0])))
        #cv2.imwrite(os.path.join('/home/uofsko-lab/Downloads/py-faster-rcnn-master/char/',format(im_name)+'-'+strCurrentChar + '.jpg'),charCrop)
        strFinalString = strFinalString + strCurrentChar

    print "\n" + strFinalString + "\n"
    cv2.imshow("imgTestingNumbers", imgTestCopy)
    #cv2.imwrite(os.path.join('/home/uofsko-lab/Documents','detect.jpg'),imgTestCopy)
    return strFinalString

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()


    prototxt = "./models/plate/test.prototxt"
    caffemodel = "./output/faster_rcnn_end2end/train/plate2_model/zf_faster_rcnn_iter_100000.caffemodel"

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['image_0124.jpg']
    #file_obj = open("./data/all/imagesets/train.txt","r")
    #im_names = file_obj.readlines()
    #print im_names
    for im_name in im_names:
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
	#im_name = im_name.replace('\r\n','.jpg')
        demo(net, im_name)
    
    plt.show()
