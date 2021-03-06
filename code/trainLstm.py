import torch
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy
import torch.optim as optim
import time
import scipy.misc
from LSTMmodel import DualLstm
from video import *
import random


def flatten_qail(_input):
    y = _input.squeeze().transpose(3, 0, 1, 2, 4)
    y = y.reshape(-1, *(y.shape[0:2])).transpose(1, 2, 0)
    return y

def get_judge():
    return nn.Sequential(OrderedDict([
        ('fc0',   nn.Linear(512,25)),
        ('sig0', nn.Sigmoid()),
        ('fc1', nn.Linear(25, 1)),
        ('sig1', nn.Sigmoid()),
                ]))

def calc_accuracy(correct, incorrect):
    correct_ = correct.cpu()
    incorrect_ = incorrect.cpu()
    return numpy.array(correct_ > incorrect_, dtype="float32").sum() / correct.shape[0]

def get_indexs(index):
    th = len(index)/2
    cIndex = []
    iIndex = []
    for i in range(len(index)):
        if index[i] < th:
            cIndex.append(i)
        else:
            iIndex.append(i)
    return cIndex, iIndex

def train(model, trk,dek, bs):
    multi = 2
    judge = get_judge()
    paras = list(model.parameters())+list(judge.parameters())
    optimizer = optim.Adam(paras, lr=0.001)
    for i in range(1):
        print("Epoch %d" % i)
        losses = []
        accs = []
        ds_size = len(trk)
        print(int(ds_size ) )
        for j in range(int(ds_size / bs) ):
            print("batch num %d" % j)
            this_trk = trk[j * bs:(j + 1) * bs]
            preloaded_train = process_data(this_trk)

            qas, visual = preloaded_train[0], preloaded_train[1]
            q, a, i = [data for data in qas]
            q = flatten_qail(q)
            a = flatten_qail(a)
            i = flatten_qail(i)
            vis_append = np.concatenate((visual,visual),axis=1).transpose(1,0,2)
            q_append = np.concatenate((q,q),axis=1).transpose(1,0,2)
            a_append = np.concatenate((a,i),axis=1).transpose(1,0,2)
            index = [k for k in range(bs*2)]
            random.shuffle(index)
            cIndex, iIndex = get_indexs(index)
            q_append = q_append[index].transpose(1,0,2)
            a_append = a_append[index].transpose(1,0,2)
            vis_append = vis_append[index].transpose(1,0,2)
            out, h = model(q_append,a_append,vis_append)
            result = judge(torch.cat((h[0][0],h[1][0]), 1))
           # print(result.shape)
            correct = result[cIndex]
            incorrect = result[iIndex]
           # print(correct, incorrect)
            correct_mean = Variable(torch.Tensor(numpy.ones((bs,1))), requires_grad=False)
            incorrect_mean = Variable(torch.Tensor(numpy.zeros((bs,1))), requires_grad=False)

            optimizer.zero_grad()

            loss = (nn.MSELoss()(correct, correct_mean.float()) + nn.MSELoss()(incorrect, incorrect_mean.float()))
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            accs.append(calc_accuracy(correct, incorrect))
            print(loss.detach())

        print("Loss %f", numpy.array(losses, dtype="float32").mean())
        print("Accs %f", numpy.array(accs, dtype="float32").mean())

        _accs = []
        ds_size = len(dek)
        for j in range(int(ds_size / bs)):
            print("batch num %d" % j)
            this_dek = dek[j * bs:(j + 1) * bs]
            preloaded_dev = process_data(this_dek)
            qas, visual = preloaded_dev[0], preloaded_dev[1]
            q, a, i = [data for data in qas]
            q = flatten_qail(q)
            a = flatten_qail(a)
            i = flatten_qail(i)
            vis_append = np.concatenate((visual, visual), axis=1).transpose(1, 0, 2)
            q_append = np.concatenate((q, q), axis=1).transpose(1, 0, 2)
            a_append = np.concatenate((a, i), axis=1).transpose(1, 0, 2)
            index = [k for k in range(bs * 2)]
            random.shuffle(index)
            cIndex, iIndex = get_indexs(index)
            q_append = q_append[index].transpose(1, 0, 2)
            a_append = a_append[index].transpose(1, 0, 2)
            vis_append = vis_append[index].transpose(1, 0, 2)
            out, h = model(q_append, a_append, vis_append)
            result = judge(torch.cat((h[0][0], h[1][0]), 1))
            correct = result[cIndex]
            incorrect = result[iIndex]
            _accs.append(calc_accuracy(correct, incorrect))
        print("Dev Accs %f", numpy.array(_accs, dtype="float32").mean())
        print("-----------")


if __name__ == "__main__":


    bs = 32
    trk, dek = get_data()
    model = DualLstm()
    train(model, trk,dek, bs)
