#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014


import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import scipy.io as sio
import cv2
import csv
from multiprocessing import Process
import random
import json
import torch

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 10
MAX_BOXES = 100


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--in', dest='infile',
                        help='input filepath',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--num_box', dest='num_box',
                        help='the number of extracted boxes, default=adaptive(10~100)',
                        default=None, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    args.prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
    args.cfg_file = 'experiments/cfgs/faster_rcnn_end2end_resnet.yml'
    args.caffemodel = 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'

    return args


def get_detections_from_im(net, im_file, conf_thresh=0.2):

    im = cv2.imread(im_file)
    if im is None:  # video stream/video file
        _, im = cv2.VideoCapture(im_file).read()

    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
   
    return {
        'image_path': im_file,                          # str
        'image_h': np.size(im, 0),                      # int
        'image_w': np.size(im, 1),                      # int
        'num_boxes' : len(keep_boxes),                  # int
        'boxes': torch.tensor(cls_boxes[keep_boxes]),   # pytorch tensor (b x 4)
        'features': torch.tensor(pool5[keep_boxes])     # pytorch tensor (b x 2048)
    }   

    
def generate_tensor(gpu_id, prototxt, weights, infile, outfile):
    assert os.path.exists(infile), '"{}" file does not exist!!'.format(infile)

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)

    result = get_detections_from_im(net, infile)

    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    torch.save(result, outfile)


     
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    random.seed(10)
    
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpu_id))

    if args.num_box is not None:
        MIN_BOXES = args.num_box
        MAX_BOXES = args.num_box

    generate_tensor(gpu_id, args.prototxt, args.caffemodel, args.infile, args.outfile)
