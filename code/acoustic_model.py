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
from mmsdk.mmmodelsdk.fusion import TensorFusion
import numpy
import pickle
from random import shuffle
import time
import numpy as np
from collections import OrderedDict
from torch.distributions.multinomial import Multinomial


print ("Tensor-MFN code for Social-IQ")
print ("Yellow warnings fro SDK are ok!")
print ("If you do happen to get nans, then the reason is the most recent acoustic features update. You can replace nans and infs in acoustic at your discretion.")


#Loading the data of Social-IQ
#Yellow warnings fro SDK are ok!
if os.path.isdir("./deployed/") is False:
        print ("Need to run the modality alignment first")
        from alignment import align,myavg
        align()
 
paths={}
paths["QA_BERT_lastlayer_binarychoice"]="./socialiq/SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE.csd"
paths["DENSENET161_1FPS"]="./deployed/SOCIAL_IQ_DENSENET161_1FPS.csd"
paths["Transcript_Raw_Chunks_BERT"]="./deployed/SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT.csd"
paths["Acoustic"]="./deployed/SOCIAL_IQ_COVAREP.csd"
social_iq=mmdatasdk.mmdataset(paths)
social_iq.unify() 




def qai_to_tensor(in_put,keys,total_i=1):
        data=dict(in_put.data)
        features=[]
        for i in range (len(keys)):
                features.append(numpy.array(data[keys[i]]["features"]))
        input_tensor=numpy.array(features,dtype="float32")[:,0,...]
        in_shape=list(input_tensor.shape)
        q_tensor=input_tensor[:,:,:,0:1,:,:]
        ai_tensor=input_tensor[:,:,:,1:,:,:]

        return q_tensor,ai_tensor[:,:,:,0:1,:,:],ai_tensor[:,:,:,1:1+total_i,:,:]


def flatten_qail(_input):
        return _input.reshape(-1,*(_input.shape[3:])).squeeze().transpose(1,0,2)
        

def build_qa_binary(qa_glove,keys):
        return qai_to_tensor(qa_glove,keys,1)


def build_visual(visual,keys):
        vis_features=[]
        for i in range (len(keys)):
                this_vis=numpy.array(visual[keys[i]]["features"])
                this_vis=numpy.concatenate([this_vis,numpy.zeros([25,2208])],axis=0)[:25,:]
                vis_features.append(this_vis)
        return numpy.array(vis_features,dtype="float32").transpose(1,0,2)

def build_acc(acoustic,keys):
        acc_features=[]
        for i in range (len(keys)):
                this_acc=numpy.array(acoustic[keys[i]]["features"])
                numpy.nan_to_num(this_acc)
                this_acc=numpy.concatenate([this_acc,numpy.zeros([25,74])],axis=0)[:25,:]
                acc_features.append(this_acc)
        final=numpy.array(acc_features,dtype="float32").transpose(1,0,2)
        return numpy.array(final,dtype="float32")

 
def build_trs(trs,keys):
        trs_features=[]
        for i in range (len(keys)):
                this_trs=numpy.array(trs[keys[i]]["features"][:,-768:])
                this_trs=numpy.concatenate([this_trs,numpy.zeros([25,768])],axis=0)[:25,:]
                trs_features.append(this_trs)
        return numpy.array(trs_features,dtype="float32").transpose(1,0,2)
 
def process_data(keys):

        qa_glove=social_iq["QA_BERT_lastlayer_binarychoice"]
        visual=social_iq["DENSENET161_1FPS"]
        transcript=social_iq["Transcript_Raw_Chunks_BERT"]
        acoustic=social_iq["Acoustic"]

        qas=build_qa_binary(qa_glove,keys)
        visual=build_visual(visual,keys)
        trs=build_trs(transcript,keys)  
        acc=build_acc(acoustic,keys)    
        
        return qas,visual,trs,acc

def to_pytorch(_input):
        return Variable(torch.tensor(_input)) 

def reshape_to_correct(_input,shape):
    return _input[:,None,None,:].expand(-1,shape[1],shape[2],-1,-1).reshape(-1,*_input.shape[-2:])
def print_model(model):
    for name, module in model.named_children():
        print(name, module)


class classifier(nn.Module):
    def __init__(self,temp_dim, input_dim, arch,input_qas_dim, qas_arch):
        super(classifier,self).__init__()
        self.temp_dim = temp_dim
        self.input_dim = input_dim
        self.input_qas_dim = input_qas_dim
        self.arch = arch
        self.qas_arch = qas_arch
        self.build()
    def build(self,):
        self.pool = nn.AvgPool2d((self.temp_dim+2,3), stride=1,padding=1)
        self.conv = nn.Conv2d(1,1,(self.temp_dim+2,3), stride=1, padding=1)
        layer = []
        input_dim = self.input_dim
        for i in range(len(self.arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,self.arch[i])))
            input_dim = self.arch[i]
        self.dock1 = nn.Sequential(OrderedDict(layer))
        self.dim = self.arch[-1]
        
        layer = []
        input_dim = self.input_qas_dim
        for i in range(len(self.qas_arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,self.qas_arch[i])))
            input_dim = self.qas_arch[i]
        self.dock2 = nn.Sequential(OrderedDict(layer))


        layer = []
        input_dim = self.input_qas_dim
        for i in range(len(self.qas_arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,self.qas_arch[i])))
            input_dim = self.qas_arch[i]
        self.dock3 = nn.Sequential(OrderedDict(layer))


        layer = []
        input_dim = self.input_qas_dim
        for i in range(len(self.qas_arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,self.qas_arch[i])))
            input_dim = self.qas_arch[i]
        self.dock4= nn.Sequential(OrderedDict(layer))

    def get_rep(self, acous, q, a, i):
        _shape = q.shape
        q_exp = torch.Tensor(flatten_qail(q)).transpose(0,1)
        a_exp = torch.Tensor(flatten_qail(a)).transpose(0,1)
        i_exp = torch.Tensor(flatten_qail(i)).transpose(0,1)
        acous_reshape = reshape_to_correct(torch.Tensor(acous.transpose(1,0,2)),_shape)
#        print("acous shape",acous.shape)
#        print("acous reshape",acous_reshape.shape)
#        print("q_exp is ",q_exp.shape)
#        print("a_exp is ",a_exp.shape)
#        print("i_exp is ",i_exp.shape)
#        print(a.shape)
        acous_reshape = acous_reshape.unsqueeze(1)
        q_exp = q_exp.unsqueeze(1)
        a_exp = a_exp.unsqueeze(1)
        i_exp = i_exp.unsqueeze(1)
        return acous_reshape,q_exp,a_exp,i_exp
        

        
        #print(pooled_acous.shape)
    def multinomial(self, prob,n):
        return torch.multinomial(prob,n,replacement=True)
    def get_multinomial(self, n,prob = torch.Tensor([1,1,1])):
        return torch.stack([Multinomial(1,prob).sample() for i in range(n)],dim = 0)
    def forward(self,acous,q,a,i):
        ac,q,a,i = self.get_rep(acous,q,a,i)
        ac_rep = self.conv(ac).squeeze()
        q_rep = self.conv(q).squeeze()
        a_rep = self.conv(a).squeeze()
        i_rep = self.conv(i).squeeze()
        print(ac_rep.shape,q_rep.shape,a_rep.shape,i_rep.shape)
        print(ac.shape,a.shape)

        weights = torch.Tensor([1/4,1/4,1/4,1/4])
        prob = self.get_multinomial(ac.shape[0],weights)
        print(prob.shape)

        

        
if __name__=="__main__":

        #if you have enough RAM, specify this as True - speeds things up ;)
        preload=False
        bs=32
        trk,dek=mmdatasdk.socialiq.standard_folds.standard_train_fold,mmdatasdk.socialiq.standard_folds.standard_valid_fold
        #This video has some issues in training set
        bads=['f5NJQiY9AuY','aHBLOkfJSYI']
        folds=[trk,dek]
        for bad in bads:
                for fold in folds:
                        try:
                                fold.remove(bad)
                        except:
                                pass

        #q_lstm,a_lstm,t_lstm,v_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn=init_tensor_mfn_modules()

        if preload is True:
                preloaded_train=process_data(trk)
                preloaded_dev=process_data(dek)
                print ("Preloading Complete")
        else:
                preloaded_data=None
        ds_size = len(trk)
        temp_dim = 25
        input_dim = 74
        input_qas_dim = 256
        qas_arch = [1024,512,256]
        arch = [128,256,512]
        model = classifier(temp_dim,input_dim, arch, input_qas_dim, qas_arch)
        print_model(model)
        for j in range(int(ds_size/bs)+1):
            this_trk = trk[j*bs:(j+1)*bs]
            preloaded_train = process_data(this_trk)
            preloaded_dev = process_data(this_trk)
            qas,visual,trs,acc = preloaded_train[0],preloaded_train[1],preloaded_train[2],preloaded_train[3]
            q,a,i = [data for data in qas]
#            print(q.shape)
#            print(acc.shape)
            model(acc, q, a, i)
            break
#            print("batch no {i}".format(i=j))
#            print("question shape is {q}, answer shape is {a}, incorrect answer shape is {i}".format(q=q.shape,a=a.shape,i=i.shape))
#            print("acoustic shape is {acous}".format(acous=acc.shape))
#            print("visual shape is {vis}".format(vis=visual.shape))
#            print("transcript shape is {trs}".format(trs=trs.shape)) #(temporal, batch, dim)
#            break


