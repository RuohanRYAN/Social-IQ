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
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


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
    def __init__(self,temp_dim, input_dim, arch,input_qas_dim, qas_arch, fuse_dim, judge_arch):
        super(classifier,self).__init__()
        self.temp_dim = temp_dim
        self.input_dim = input_dim
        self.input_qas_dim = input_qas_dim
        self.arch = arch
        self.qas_arch = qas_arch
        self.fuse_dim = fuse_dim
        self.judge_arch = judge_arch
        self.build()
        
    def build(self,):
        self.pool = nn.AvgPool2d((self.temp_dim+2,3), stride=1,padding=1)
        # self.conv_a = nn.Conv2d(768,768,(self.temp_dim+2,3), stride=1, padding=1)
        # self.conv_i = nn.Conv2d(576,576,(self.temp_dim+2,3), stride=1, padding=1)
        self.conv = nn.Conv2d(1,1,(self.temp_dim+2,3), stride=1, padding=1)
        layer = []
        input_dim = self.input_dim
        for i in range(len(self.arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,self.arch[i])))
            layer.append(("Non-linear{i}".format(i=i), nn.ReLU()))
            input_dim = self.arch[i]
        self.dock1 = nn.Sequential(OrderedDict(layer))
        self.dim = self.arch[-1]
        
        layer = []
        input_dim = self.input_qas_dim
        for i in range(len(self.qas_arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,self.qas_arch[i])))
            layer.append(("Non-linear{i}".format(i=i), nn.ReLU()))
            input_dim = self.qas_arch[i]
        self.dock2 = nn.Sequential(OrderedDict(layer))


        layer = []
        input_dim = self.input_qas_dim
        for i in range(len(self.qas_arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,self.qas_arch[i])))
            layer.append(("Non-linear{i}".format(i=i), nn.ReLU()))
            input_dim = self.qas_arch[i]
        self.dock3 = nn.Sequential(OrderedDict(layer))


        layer = []
        input_dim = self.input_qas_dim
        for i in range(len(self.qas_arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,self.qas_arch[i])))
            layer.append(("Non-linear{i}".format(i=i), nn.ReLU()))
            input_dim = self.qas_arch[i]
        self.dock4= nn.Sequential(OrderedDict(layer))

        layer = []
        input_dim = self.fuse_dim
        for i in range(len(self.judge_arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,self.judge_arch[i])))
            if(i!=len(self.judge_arch)-1):
                layer.append(("Non-linear{i}".format(i=i), nn.ReLU()))
            
            input_dim = self.judge_arch[i]
        #layer.append(("final layer", nn.LogSigmoid()))
        layer.append(("final layer", nn.LogSoftmax(dim=1)))
        self.judge = nn.Sequential(OrderedDict(layer))
        

    def get_rep(self, acous, q, a, i):
        _shape = q.shape
        q_exp = torch.Tensor(flatten_qail(q)).transpose(0,1)
        a_exp = torch.Tensor(flatten_qail(a)).transpose(0,1)
        i_exp = torch.Tensor(flatten_qail(i)).transpose(0,1)
        acous_reshape = reshape_to_correct(torch.Tensor(acous.transpose(1,0,2)),_shape)
        acous_reshape = acous_reshape.unsqueeze(1)
        q_exp = q_exp.unsqueeze(1)
        a_exp = a_exp.unsqueeze(1)
        i_exp = i_exp.unsqueeze(1)
        return acous_reshape,q_exp,a_exp,i_exp
        

        
    def multinomial(self, prob,n):
        return torch.multinomial(prob,n,replacement=True)
    def get_multinomial(self, n,prob = torch.Tensor([1,1,1])):
        return torch.stack([Multinomial(1,prob).sample() for i in range(n)],dim = 0)
    def forward(self,acous,a,i):
        # print(acous.shape, a.shape, i.shape)
        # print(type(acous),type(a),type(i))
        acous = torch.tensor(acous).transpose(1,0)
        acous_a = acous[:,None,:,:].expand(-1,a.shape[1],-1,-1)
        acous_a = acous_a.reshape(-1,*acous_a.shape[2:])

        acous_i = acous[:,None,:,:].expand(-1,i.shape[1],-1,-1)
        acous_i = acous_i.reshape(-1,*acous_i.shape[2:])
        a = torch.tensor(a)
        i = torch.tensor(i)
        # convolutional layer to reduce the temporal dim 
        acous_a_rep = self.conv(acous_a[:,None,:,:]).squeeze()
        acous_i_rep = self.conv(acous_i[:,None,:,:]).squeeze()

        a = a.reshape(-1,a.shape[-1])
        i = i.reshape(-1,i.shape[-1])

        acous_a_doc = self.dock1(acous_a_rep)
        acous_i_doc = self.dock1(acous_i_rep)

        a_doc = self.dock2(a)
        i_doc = self.dock3(i)
        # print(acous_a_rep.shape)
        # print(acous_i_rep.shape)
        # print(a.shape)
        # print(i.shape)
        # print(acous_a_doc.shape)
        # print(acous_i_doc.shape)
        # print(a_doc.shape)
        # print(i_doc.shape)

        fuse_a = torch.cat((acous_a_doc, a_doc),dim=1)
        fuse_i = torch.cat((acous_i_doc, i_doc),dim=1)
        a_res = self.judge(fuse_a)
        i_res = self.judge(fuse_i)
        # print("--------")
        # print(fuse_a.shape)
        # print(fuse_i.shape)
        # print(a_res.shape)
        # print(i_res.shape)
        return a_res, i_res
    def predict(self,acous,a,i):
        a_res, i_res = self.forward(acous,a,i)
        a_res_reshape = torch.argmax(a_res,dim=1,keepdim=True).reshape(*(6,4),-1)
        i_res_reshape = torch.argmax(i_res,dim=1,keepdim=True).reshape(*(6,3),-1)
#        print(a_res.shape,i_res.shape)
        print(a_res)
        print(i_res)
        return a_res, i_res, a_res_reshape, i_res_reshape

        

#    def forward(self,acous,q,a,i):
#        ac,q,a,i = self.get_rep(acous,q,a,i)
#        print(q.shape)
#        ac_rep = self.conv(ac).squeeze()
#        q_rep = self.conv(q).squeeze()
#        a_rep = self.conv(a).squeeze()
#        i_rep = self.conv(i).squeeze()
#        #print(ac_rep.shape,q_rep.shape,a_rep.shape,i_rep.shape)
#        #print(ac.shape,a.shape)
#
#        weights = torch.Tensor([1/4,1/4,1/4,1/4])
#        prob = self.get_multinomial(ac.shape[0],weights)
#        ac_dock = self.dock1(ac_rep)
#        q_dock = self.dock2(q_rep)
#        a_dock = self.dock3(a_rep)
#        i_dock = self.dock4(i_rep)
#
#        #print(ac_dock.shape,q_dock.shape,q_dock.shape,i_dock.shape)
#
#        fuse_a = torch.cat((ac_dock,q_dock,a_dock),dim=1)
#        fuse_i = torch.cat((ac_dock,q_dock,i_dock),dim=1)
#        
#        #print(fuse_a.shape, fuse_i.shape)
#        a_res = self.judge(fuse_a) 
#        i_res = self.judge(fuse_i)
#        return a_res,i_res
def contains_nan(x):
    a = np.isnan(x)
    b = x==float('inf')
    c = x==float('-inf')
    return not np.sum(a,dtype=int)==0 ,np.any(b) ,np.any(c)
def convert_nans(x):
    nan,inf,n_inf = contains_nan(x)
    if(nan or inf or n_inf):
        return np.nan_to_num(x,posinf=20,neginf=0)
    return x 
def load_qai(trk):
    path = "/home/gaoruohan19/project/Social-IQ-new/data/qai/"

    a_arr = []
    i_arr = []
    for folder in trk:
        for file in os.listdir(path+folder+"_trimmed"):
            mat_path = path+folder+"_trimmed/"+file
            array = np.load(mat_path)
            # print(mat_path)
            # print(array.shape)
            if (file[0] == "a"):
                a_arr.append(array)
                # print(array.shape)
                if(array.shape[0]!=24): print(folder)
            else:
                i_arr.append(array)
                # print(array.shape)
                if(array.shape[0]!=18): print(folder)
    
    a_arr = np.stack(a_arr,axis=0)
    i_arr = np.stack(i_arr,axis=0)
    # print(a_arr.shape)
    # print(i_arr.shape)
    return a_arr,i_arr

if __name__=="__main__":

        #if you have enough RAM, specify this as True - speeds things up ;)
        preload=False
        bs=32
        trk,dek=mmdatasdk.socialiq.standard_folds.standard_train_fold,mmdatasdk.socialiq.standard_folds.standard_valid_fold
        #This video has some issues in training set
        bads=['f5NJQiY9AuY','aHBLOkfJSYI','OWsT2rkqCk8','GxYimeaoea0','o4CKGkaTn-A','gbVOyKifrAo','FiLWlqVE9Fg','srWtQnseRyE','_UJNNySGM6Q','N-6zVmVuTs0','gBs-CkxGXy8','j1CTHVQ8Z3k','ggLOXOiq7WE','2ihOXaU0I8o']
        folds=[trk,dek]
        for bad in bads:
                for fold in folds:
                        try:
                                fold.remove(bad)
                        except:
                                pass


        if preload is True:
                preloaded_train=process_data(trk)
                preloaded_dev=process_data(dek)
                print ("Preloading Complete")
        else:
                preloaded_data=None
        ds_size = len(trk)
        temp_dim = 25
        input_dim = 74
        input_qas_dim = 768
        qas_arch = [1024,512,256]
        arch = [128,256,512]
        judge_arch = [512,256,64,2]
        fuse_dim = 768
        model = classifier(temp_dim,input_dim, arch, input_qas_dim, qas_arch, fuse_dim, judge_arch)
        optimizer = torch.optim.Adam(model.parameters(),weight_decay = 0.0)
        loss = nn.NLLLoss()
        
        print_model(model)
        for j in range(int(ds_size/bs)+1):
            this_trk = trk[j*bs:(j+1)*bs]
            preloaded_train = process_data(this_trk)
            #preloaded_dev = process_data(this_trk)
            qas,_,_,acc = preloaded_train[0],preloaded_train[1],preloaded_train[2],preloaded_train[3]
            a_arr, i_arr = load_qai(this_trk)
            #print(a_arr.shape, i_arr.shape)
            #print(acc.shape)
            is_nan = contains_nan(acc)
            print("there is nan in acoustic {}".format(is_nan))
            # if(True in is_nan):
            #     continue
            q,a,i = [data for data in qas]
            acc = convert_nans(acc)
#            print(q.shape,a.shape,i.shape)
            a_res, i_res = model(acc,a_arr, i_arr)
#            a_res,i_res = model(acc, q, a, i)
            true = torch.ones(a_res.shape[0],dtype=torch.long)
            false = torch.zeros(i_res.shape[0],dtype=torch.long)
            loss_a = loss(a_res, true)
            loss_i = loss(i_res, false)
            loss_tot = loss_a+loss_i

            loss_tot.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("total loss is {loss_tot},loss_a is {loss_a},loss_i is {loss_i}".format(loss_tot=loss_tot,loss_a=loss_a,loss_i=loss_i))
            
            with torch.no_grad():
                a_pred = torch.argmax(a_res,dim=1)
                i_pred = torch.argmax(i_res,dim=1)
                pred = torch.cat((a_pred,i_pred),dim=0).numpy()
                ground_truth = torch.cat((true,false),dim=0).numpy()
                f1 = f1_score(ground_truth, pred)
                print("f1 score is {}".format(f1))
                
            print("________")
            # break
        print("finish epoch {j}".format(j=j))
        ## validation ##
        with torch.no_grad():
            ds_size = len(dek)
            bs = 1
            for i in range(int(ds_size/bs)+1):
                this_dek = dek[i*bs:(i+1)*bs]
                #print("validation batches")
                print(this_dek) 
                if(len(this_dek)==0): continue 
                preload_dev = process_data(this_dek)
                qas,_,_,acc = preload_dev[0],preload_dev[1],preload_dev[2],preload_dev[3]
                a_arr, i_arr = load_qai(this_dek)
                if(a_arr.shape[1]!=24 or i_arr.shape[1]!=18): 
                    print(a_arr.shape,i_arr.shape)
                    print(this_dek)

                a_res, i_res, a_res_reshape, i_res_reshape = model.predict(acc,a_arr,i_arr)
                break
                #is_nan = contains_nan(acc)
#                q,a,i=[data for data in qas]
#                acc = convert_nans(acc)
#                a_res_dev, i_res_dev = model(acc,q,a,i)
#                a_res = torch.argmax(a_res_dev,dim=1)
#                i_res = torch.argmax(i_res_dev,dim=1)
#                print(a_res)
#                print(torch.sum(a_res))
#                print(torch.sum(i_res))
#                break


