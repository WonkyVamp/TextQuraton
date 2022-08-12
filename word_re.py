import difflib
import importlib
import random
import string

from tqdm import tqdm
import numpy as np
random.seed(123)
from ocr.utils.sclite_helper import ScliteHelper

from ocr.utils.expand_bounding_box import expand_bounding_box
from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, decode_char, EOS, BOS
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

from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding



test_ds = IAMDataset("form_original", train=False)
random.seed(1)
figs_to_plot = 4
images = []

word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
word_segmentation_net.load_parameters("models/word_segmentation2.params")
word_segmentation_net.hybridize()


min_c = 0.1
overlap_thres = 0.1
topk = 600

fig, test_axs = plt.subplots(int(len(p_s_img)/2), 2, 
                        figsize=(15, 5 * int(len(p_s_img)/2)))
predicted_words_bbs_array = []

for i, p_s_img_2 in enumerate(p_s_img):
    s_y, s_x = int(i/2), int(i%2)

    predicted_bb = predict_bounding_boxes(
        word_segmentation_net, p_s_img_2, min_c, overlap_thres, topk, ctx)

    predicted_words_bbs_array.append(predicted_bb)
    
    test_axs[s_y, s_x].imshow(p_s_img_2, cmap='Greys_r')
    for j in range(predicted_bb.shape[0]):     
        (x, y, w, h) = predicted_bb[j]
        ht, wt = p_s_img_2.shape[-2:]
        (x, y, w, h) = (x * wt, y * ht, w * wt, h * ht)
        rect = patches.Rectangle((x, y), w, h, fill=False, color="r")
        test_axs[s_y, s_x].add_patch(rect)
        test_axs[s_y, s_x].axis('off')




handwriting_line_recognition_net = HandwritingRecognitionNet(rnn_hidden_states=512,
                                                             rnn_layers=2, ctx=ctx, max_seq_len=160)
handwriting_line_recognition_net.load_parameters("models/handwriting_line8.params", ctx=ctx)
handwriting_line_recognition_net.hybridize()


line_image_size = (60, 800)
character_probs = []
for line_images in line_images_array:
    form_character_prob = []
    for i, line_image in enumerate(line_images):
        line_image = handwriting_recognition_transform(line_image, line_image_size)
        line_character_prob = handwriting_line_recognition_net(line_image.as_in_context(ctx))
        form_character_prob.append(line_character_prob)
    character_probs.append(form_character_prob)




def get_arg_max(prob):
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]


def get_beam_search(prob, width=5):
    possibilities = ctcBeamSearch(prob.softmax()[0].asnumpy(), alphabet_encoding, None, width)
    return possibilities[0]


FEATURE_LEN = 150
denoiser = Denoiser(alphabet_size=len(ALPHABET), max_src_length=FEATURE_LEN, max_tgt_length=FEATURE_LEN, num_heads=16, embed_size=256, num_layers=2)
denoiser.load_parameters('models/denoiser2.params', ctx=ctx)
denoiser.hybridize(static_alloc=True)



ctx_nlp = mx.gpu(3)
language_model, vocab = nlp.model.big_rnn_lm_2048_512(dataset_name='gbw', pretrained=True, ctx=ctx_nlp)
moses_tokenizer = nlp.data.SacreMosesTokenizer()
moses_detokenizer = nlp.data.SacreMosesDetokenizer()

beam_sampler = nlp.model.BeamSearchSampler(beam_size=20,
                                           decoder=denoiser.decode_logprob,
                                           eos_id=EOS,
                                           scorer=nlp.model.BeamSearchScorer(),
                                           max_length=150)

generator = SequenceGenerator(beam_sampler, language_model, vocab, ctx_nlp, moses_tokenizer, moses_detokenizer)
def get_denoised(prob, ctc_bs=False):
    if ctc_bs: 
        text = get_beam_search(prob)
    else:
        text = get_arg_max(prob)
    src_seq, src_valid_length = encode_char(text)
    src_seq = mx.nd.array([src_seq], ctx=ctx)
    src_valid_length = mx.nd.array(src_valid_length, ctx=ctx)
    encoder_outputs, _ = denoiser.encode(src_seq, valid_length=src_valid_length)
    states = denoiser.decoder.init_state_from_encoder(encoder_outputs, 
                                                      encoder_valid_length=src_valid_length)
    inputs = mx.nd.full(shape=(1,), ctx=src_seq.context, dtype=np.float32, val=BOS)
    output = generator.generate_sequences(inputs, states, text)
    return output.strip()
# testing
# GG works!
#
# sentence = "This sentnce has an eror"
# src_seq, src_valid_length = encode_char(sentence)
# src_seq = mx.nd.array([src_seq], ctx=ctx)
# src_valid_length = mx.nd.array(src_valid_length, ctx=ctx)
# encoder_outputs, _ = denoiser.encode(src_seq, valid_length=src_valid_length)
# states = denoiser.decoder.init_state_from_encoder(encoder_outputs, 
#                                                   encoder_valid_length=src_valid_length)
# inputs = mx.nd.full(shape=(1,), ctx=src_seq.context, dtype=np.float32, val=BOS)
# print(sentence)
# print("Choice")
print(generator.generate_sequences(inputs, states, sentence))
for i, form_character_probs in enumerate(character_probs):
    fig, test_axs = plt.subplots(len(form_character_probs) + 1, 
                            figsize=(10, int(1 + 2.3 * len(form_character_probs))))
    for j, line_character_probs in enumerate(form_character_probs):
        decoded_line_am = get_arg_max(line_character_probs)
        print("[AM]",decoded_line_am)
        decoded_line_bs = get_beam_search(line_character_probs)
        decoded_line_denoiser = get_denoised(line_character_probs, ctc_bs=False)
        print("[D ]",decoded_line_denoiser)
        
        line_image = line_images_array[i][j]
        test_axs[j].imshow(line_image.squeeze(), cmap='Greys_r')            
        test_axs[j].set_title("[AM]: {}\n[BS]: {}\n[D ]: {}\n\n".format(decoded_line_am, decoded_line_bs, decoded_line_denoiser), fontdict={"horizontalalignment":"left", "family":"monospace"}, x=0)
        test_axs[j].axis('off')
    test_axs[-1].imshow(np.zeros(shape=line_image_size), cmap='Greys_r')
    test_axs[-1].axis('off')

