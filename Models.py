
# In the rest of the tutorial, we assume that the `plt`
# is imported before every code snippet.
import chainer
from chainercv import utils

import matplotlib.pyplot as plt

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv2
from chainercv.links import YOLOv3
from chainercv import utils
from chainercv.visualizations import vis_bbox

import timeit
import time

def runRCNN():
    startTime = time.time()
    #print(startTime)
    # Read an RGB image and return it in CHW format.
    img = read_image('sample1.jpg')
    model = SSD300(pretrained_model='voc0712')
    model1 = SSD512(pretrained_model='voc0712')
    bboxes, labels, scores = model1.predict([img])
    vis_bbox(img, bboxes[0], labels[0], scores[0],
             label_names=voc_bbox_label_names)
    #print(time.time() - startTime)
    #plt.show()

def runSSD():
    aimage = 'sample1.jpg'
    amodel = 'ssd300'
    apretrained_model = 'voc0712'
    agpu = -1

    if amodel == 'ssd300':
        model = SSD300(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=apretrained_model)
    elif amodel == 'ssd512':
        model = SSD512(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=apretrained_model)

    if agpu >= 0:
        chainer.cuda.get_device_from_id(agpu).use()
        model.to_gpu()

    img = utils.read_image(aimage, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=voc_bbox_label_names)
    plt.show()

def runYOLO():
    '''parser = argparse.ArgumentParser()
      parser.add_argument(
          '--model', choices=('yolo_v2', 'yolo_v3'),
          default='yolo_v2')
      parser.add_argument('--gpu', type=int, default=-1)
      parser.add_argument('--pretrained-model', default='voc0712')
      parser.add_argument('image')
      args = parser.parse_args()'''

    aimage = 'living_room.jpeg'
    amodel = 'yolo_v2'
    apretrained_model = 'voc0712'
    agpu = -1

    if amodel == 'yolo_v2':
        model = YOLOv2(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=apretrained_model)
    elif amodel == 'yolo_v3':
        model = YOLOv3(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=apretrained_model)

    if agpu >= 0:
        chainer.cuda.get_device_from_id(agpu).use()
        model.to_gpu()

    img = utils.read_image(aimage, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=voc_bbox_label_names)
    plt.show()


if __name__ == '__main__':
    print("RUNNING")
    totalTime = timeit.timeit("runRCNN()", setup="from __main__ import runRCNN",number=10)
    print(totalTime/10)
