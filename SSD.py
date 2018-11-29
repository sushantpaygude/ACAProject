import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
   '''parser = argparse.ArgumentParser()
   parser.add_argument(
       '--model', choices=('ssd300', 'ssd512'), default='ssd300')
   parser.add_argument('--gpu', type=int, default=-1)
   parser.add_argument('--pretrained-model', default='voc0712')
   parser.add_argument('image')
   args = parser.parse_args()'''
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

main()