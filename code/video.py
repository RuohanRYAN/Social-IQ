import torch
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy
import torch.optim as optim
import time
import scipy.misc
import os
from model import mylstm
import h5py
import mmsdk
from mmsdk import mmdatasdk

import numpy
import pickle
from random import shuffle
import time
import numpy as np

# Loading the data of Social-IQ
# Yellow warnings fro SDK are ok!
if os.path.isdir("./deployed/") is False:
    print("Need to run the modality alignment first")
    from alignment import align, myavg

    align()

paths = {}
paths["QA_BERT_lastlayer_binarychoice"] = "./socialiq/SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE.csd"
paths["DENSENET161_1FPS"] = "./deployed/SOCIAL_IQ_DENSENET161_1FPS.csd"
paths["Transcript_Raw_Chunks_BERT"] = "./deployed/SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT.csd"
paths["Acoustic"] = "./deployed/SOCIAL_IQ_COVAREP.csd"
social_iq = mmdatasdk.mmdataset(paths)
social_iq.unify()


def qai_to_tensor(in_put, keys, total_i=1):
    data = dict(in_put.data)
    features = []
    for i in range(len(keys)):
        features.append(numpy.array(data[keys[i]]["features"]))
    input_tensor = numpy.array(features, dtype="float32")[:, 0, ...]
    in_shape = list(input_tensor.shape)
    q_tensor = input_tensor[:, :, :, 0:1, :, :]
    ai_tensor = input_tensor[:, :, :, 1:, :, :]

    return q_tensor, ai_tensor[:, :, :, 0:1, :, :], ai_tensor[:, :, :, 1:1 + total_i, :, :]


def flatten_qail(_input):
    return _input.reshape(-1, *(_input.shape[3:])).squeeze().transpose(1, 0, 2)


def build_qa_binary(qa_glove, keys):
    return qai_to_tensor(qa_glove, keys, 1)


def build_visual(visual, keys):
    vis_features = []
    for i in range(len(keys)):
        this_vis = numpy.array(visual[keys[i]]["features"])
        this_vis = numpy.concatenate([this_vis, numpy.zeros([25, 2208])], axis=0)[:25, :]
        vis_features.append(this_vis)
    return numpy.array(vis_features, dtype="float32").transpose(1, 0, 2)


def build_acc(acoustic, keys):
    acc_features = []
    for i in range(len(keys)):
        this_acc = numpy.array(acoustic[keys[i]]["features"])
        numpy.nan_to_num(this_acc)
        this_acc = numpy.concatenate([this_acc, numpy.zeros([25, 74])], axis=0)[:25, :]
        acc_features.append(this_acc)
    final = numpy.array(acc_features, dtype="float32").transpose(1, 0, 2)
    return numpy.array(final, dtype="float32")


def build_trs(trs, keys):
    trs_features = []
    for i in range(len(keys)):
        this_trs = numpy.array(trs[keys[i]]["features"][:, -768:])
        this_trs = numpy.concatenate([this_trs, numpy.zeros([25, 768])], axis=0)[:25, :]
        trs_features.append(this_trs)
    return numpy.array(trs_features, dtype="float32").transpose(1, 0, 2)


def process_data(keys):
    qa_glove = social_iq["QA_BERT_lastlayer_binarychoice"]
    visual = social_iq["DENSENET161_1FPS"]
    transcript = social_iq["Transcript_Raw_Chunks_BERT"]
    acoustic = social_iq["Acoustic"]

    qas = build_qa_binary(qa_glove, keys)
    visual = build_visual(visual, keys)
    trs = build_trs(transcript, keys)
    acc = build_acc(acoustic, keys)

    return qas, visual, trs, acc


def to_pytorch(_input):
    return Variable(torch.tensor(_input))




def reshape_to_correct(_input, shape):
    return _input[:, None, None, :].expand(-1, shape[1], shape[2], -1).reshape(-1, _input.shape[1])

def get_data():
    trk, dek = mmdatasdk.socialiq.standard_folds.standard_train_fold, mmdatasdk.socialiq.standard_folds.standard_valid_fold
    # This video has some issues in training set
    bads = ['f5NJQiY9AuY', 'aHBLOkfJSYI']
    folds = [trk, dek]
    for bad in bads:
        for fold in folds:
            try:
                fold.remove(bad)
            except:
                pass
    return trk, dek




