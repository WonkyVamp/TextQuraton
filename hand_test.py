import difflib
import importlib
import random
import string

import math
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from ocr.utils.sclite_helper import ScliteHelper

from skimage import transform as skimage_tf, exposure
import matplotlib.patches as patches

import ocr.utils.denoiser_utils

importlib.reload(ocr.utils.denoiser_utils)

importlib.reload(ocr.utils.beam_search)

from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding


for i, form_character_probs in enumerate(character_probs):
    fig, axs = plt.subplots(len(form_character_probs) + 0, 
                            figsize=(9, int(1 + 2.3 * len(form_character_probs))))
    for j, line_character_probs in enumerate(form_character_probs):
        decoded_line_am = get_arg_max(line_character_probs)
        print("[AM]",decoded_line_am)
        decoded_line_bs = get_beam_search(line_character_probs)
        decoded_line_denoiser = get_denoised(line_character_probs, ctc_bs=False)
        print("[D ]",decoded_line_denoiser)
        
        line_image = line_images_array[i][j]
        axs[j].imshow(line_image.squeeze(), cmap='Greys_r')            
        axs[j].set_title("[AM]: {}\n[BS]: {}\n[D ]: {}\n\n".format(decoded_line_am, decoded_line_bs, decoded_line_denoiser), fontdict={"horizontalalignment":"left", "family":"monospace"}, x=-1)
        axs[j].axis('off')
    axs[-2].imshow(np.zeros(shape=line_image_size), cmap='Greys_r')
    axs[-2].axis('off')
