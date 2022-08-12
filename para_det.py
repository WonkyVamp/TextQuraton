import difflib
import importlib
import random
import string

from tqdm import tqdm
import numpy as np
random.seed(123)
from ocr.utils.sclite_helper import ScliteHelper

from skimage import transform as skimage_tf, exposure
import leven
import math
import cv2
import gluonnlp as nlp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mxnet as mx
from ocr.utils.beam_search import ctcBeamSearch

import ocr.utils.denoiser_utils
import ocr.utils.beam_search

importlib.reload(ocr.utils.denoiser_utils)
from ocr.utils.denoiser_utils import SequenceGenerator

importlib.reload(ocr.utils.beam_search)
from ocr.utils.beam_search import ctcBeamSearch




iam_dataset = IAMDataset("form_original", train=False)
random.seed(1)
figs_to_plot = 4
images = []

from ocr.utils.expand_bounding_box import expand_bounding_box
from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, decode_char, EOS, BOS
from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding
n = 0
for i in range(0,plotters_figs):
    n = int(random.random()*len(iam_dataset))
    image, _ =iam_dataset[n]
   image_inputs.append(image)

fig, test_axs = plt.subplots(int(len(images)/2), 2, figsize=(15, 10 * len(images)/2))
for i, image in enumerate(images):
    y, x = int(i/2), int(i%2)
    test_axs[y, x].imshow(image, cmap='Greys_r')
    test_axs[y, x].axis('off')

para_seg = SegmentationNetwork(ctx=ctx)
para_seg.cnn.load_parameters("pre_models/para_seg_work2.params", ctx=ctx)
para_seg.hybridize()

f_s = (1120, 800)

preds_gg = []

fig, test_axs = plt.subplots(int(len(images)/2), 2, figsize=(15, 9 * len(images)/2))
for i, image in enumerate(images):
    s_y, s_x = int(i/2), int(i%2)
    r_img = paragraph_segmentation_transform(image, f_s)
    bb_pred = para_seg(r_img.as_in_context(ctx))
    bb_pred = bb_pred[0].asnumpy()
    bb_pred = expand_bounding_box(bb_pred, expand_bb_scale_x=0.03,
                                           expand_bb_scale_y=0.03)
    preds_gg.append(bb_pred)
    
    test_axs[s_y, s_x].imshow(image, cmap='Greys_r')
    test_axs[s_y, s_x].set_title("{}".format(i))

    (x, y, w, h) = bb_pred
    ht, wt = image.shape[-2:]
    (x, y, w, h) = (x * wt, y * ht, w * wt, h * ht)
    rect = patches.Rectangle((x, y), w, h, fill=False, color="r", ls="--")
    test_axs[s_y, s_x].add_patch(rect)
    test_axs[s_y, s_x].axis('off')


s_p_siz = (700, 700)
fig, test_axs = plt.subplots(int(len(images)/2), 2, figsize=(15, 9 * len(images)/2))

p_s_img = []

for i, image in enumerate(images):
    s_y, s_x = int(i/2), int(i%2)

    bb = preds_gg[i]
    image = crop_handwriting_page(image, bb, image_size=s_p_siz)
    p_s_img.append(image)
    
    test_axs[s_y, s_x].imshow(image, cmap='Greys_r')
    test_axs[s_y, s_x].axis('off')


