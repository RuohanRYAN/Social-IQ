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
    def __init__(self,temp_dim,acous_input_dim,vis_input_dim,trs_input_dim,arch,vis_arch,trs_arch,input_qas_dim,qas_arch,fuse_dim,judge_arch):
        super(classifier,self).__init__()
        self.temp_dim = temp_dim
        self.acous_input_dim = acous_input_dim
        self.vis_input_dim = vis_input_dim
        self.trs_input_dim = trs_input_dim
        self.input_qas_dim = input_qas_dim

        self.arch = arch
        self.vis_arch = vis_arch
        self.trs_arch = trs_arch 
        self.qas_arch = qas_arch
        # self.fuse_dim = self.arch[-1]+self.vis_arch[-1]+self.trs_arch[-1]+qas_arch[-1]
        self.fuse_dim = 100+self.qas_arch[-1]
        self.judge_arch = judge_arch

        self.acous_hidden_size = self.acous_input_dim
        self.vis_hidden_size = self.vis_input_dim // 4
        self.trs_hidden_size = self.trs_input_dim // 4
        self.hidden_qas_size = 128
        self.build()
        
    def build(self,):
        # self.pool = nn.AvgPool2d((self.temp_dim+2,3), stride=1,padding=1)
        # self.conv = nn.Conv2d(1,1,(self.temp_dim+2,3), stride=1, padding=1)


        ## initialize lstm for acoustic features 
        self.lstm_acous = nn.LSTM(self.acous_input_dim,self.acous_hidden_size,1,batch_first=True)
        ## initialize lstm for visual features 
        self.lstm_vis = nn.LSTM(self.vis_input_dim,self.vis_hidden_size,1,batch_first=True)
        ## initialize lstm for trs features 
        self.lstm_trs = nn.LSTM(self.trs_input_dim,self.trs_hidden_size,1,batch_first=True)
        ## initialize lstm for qa features
        self.lstm_qa = nn.LSTM(self.input_qas_dim,self.hidden_qas_size,1,batch_first=True)


        self.dock1 = self.build_dock(self.arch,self.acous_hidden_size)
        self.dock2 = self.build_dock(self.qas_arch,self.hidden_qas_size)
        self.dock3 = self.build_dock(self.vis_arch,self.vis_hidden_size)
        self.dock4 = self.build_dock(self.trs_arch,self.trs_hidden_size)


        layer = []
        input_dim = self.fuse_dim
        for i in range(len(self.judge_arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,self.judge_arch[i])))
            if(i!=len(self.judge_arch)-1):
                layer.append(("Non-linear{i}".format(i=i), nn.ReLU()))
            
            input_dim = self.judge_arch[i]
        layer.append(("final layer", nn.LogSoftmax(dim=1)))
        self.judge = nn.Sequential(OrderedDict(layer))
    
    def build_dock(self,arch,input_dim):
        layer = []
        input_dim = input_dim
        for i in range(len(arch)):
            layer.append(("linear {i}".format(i=i), nn.Linear(input_dim,arch[i])))
            layer.append(("Non-linear{i}".format(i=i), nn.ReLU()))
            input_dim = arch[i]
        dock = nn.Sequential(OrderedDict(layer))
        return dock

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
    def transform(self,acous,a,i):
        acous = torch.tensor(acous).transpose(1,0)
        acous_a = acous[:,None,:,:].expand(-1,a.shape[1],-1,-1)
        acous_a = acous_a.reshape(-1,*acous_a.shape[2:])

        acous_i = acous[:,None,:,:].expand(-1,i.shape[1],-1,-1)
        acous_i = acous_i.reshape(-1,*acous_i.shape[2:])
        return acous_a, acous_i
    def forward_lstm(self,model,acous_a,acous_i,acous_hidden_size):
        a_h0 = torch.randn(1,acous_a.shape[0],acous_hidden_size)
        a_c0 = torch.randn(1,acous_a.shape[0],acous_hidden_size)
        i_h0 = torch.randn(1,acous_i.shape[0],acous_hidden_size)
        i_c0 = torch.randn(1,acous_i.shape[0],acous_hidden_size)
        _,(ahn,acn) = model(acous_a,(a_h0,a_c0))
        _,(ihn,icn) = model(acous_i,(i_h0,i_c0))
        return ahn,ihn
    def forward(self,acous,vis,trs,a,i):
        acous_a, acous_i = self.transform(acous,a,i)
        vis_a, vis_i =self.transform(vis,a,i)
        trs_a, trs_i = self.transform(trs,a,i)
        # print(acous_a.shape,vis_a.shape,trs_a.shape)
        a = torch.tensor(a)
        i = torch.tensor(i)
        a = a.reshape(-1,*a.shape[2:])
        i = i.reshape(-1,*i.shape[2:])



        self.qa_h0 = torch.randn(1,acous_a.shape[0],self.hidden_qas_size)
        self.qa_c0 = torch.randn(1,acous_a.shape[0],self.hidden_qas_size)
        self.qi_h0 = torch.randn(1,acous_i.shape[0],self.hidden_qas_size)
        self.qi_c0 = torch.randn(1,acous_i.shape[0],self.hidden_qas_size)

        acous_ahn, acous_ihn = self.forward_lstm(self.lstm_acous,acous_a,acous_i,self.acous_hidden_size)
        vis_ahn, vis_ihn = self.forward_lstm(self.lstm_vis,vis_a,vis_i,self.vis_hidden_size)
        trs_ahn, trs_ihn = self.forward_lstm(self.lstm_trs,trs_a,trs_i,self.trs_hidden_size)

        _,(a_doc_hn,a_doc_cn) = self.lstm_qa(a,(self.qa_h0,self.qa_c0))
        _,(i_doc_hn,i_doc_cn) = self.lstm_qa(i,(self.qi_h0,self.qi_c0))

        acous_a_doc = self.dock1(acous_ahn.squeeze())
        acous_i_doc = self.dock1(acous_ihn.squeeze())

        vis_a_doc = self.dock3(vis_ahn.squeeze())
        vis_i_doc = self.dock3(vis_ihn.squeeze())

        trs_a_doc = self.dock4(trs_ahn.squeeze())
        trs_i_doc = self.dock4(trs_ihn.squeeze())


        a_doc = self.dock2(a_doc_hn.squeeze())
        i_doc = self.dock2(i_doc_hn.squeeze())

        ### tensor fusion model 
        fmodel = TensorFusion([acous_a_doc.shape[-1],vis_a_doc.shape[-1],trs_a_doc.shape[-1]],self.fuse_dim-self.qas_arch[-1])
        fa_result = fmodel([acous_a_doc,vis_a_doc,trs_a_doc])
        fi_result = fmodel([acous_i_doc,vis_i_doc,trs_i_doc])
        fuse_a = torch.cat((fa_result, a_doc),dim=1)
        fuse_i = torch.cat((fi_result, i_doc),dim=1)
        a_res = self.judge(fuse_a)
        i_res = self.judge(fuse_i)
        # print(fa_result.shape)
        return a_res ,i_res 
        # print(acous_a_doc.shape, vis_a_doc.shape, trs_a_doc.shape,a_doc.shape)
        # fuse_a = torch.cat((acous_a_doc, vis_a_doc,trs_a_doc, a_doc),dim=1)
        # fuse_i = torch.cat((acous_i_doc, vis_i_doc,trs_i_doc, i_doc),dim=1)
        # a_res = self.judge(fuse_a)
        # i_res = self.judge(fuse_i)

        # return a_res, i_res
    def predict(self,acous,a,i):
        # a_res, i_res = self.forward(acous,a,i)
        print("original acoustic shape is {}".format(acous.shape))
        acous = torch.tensor(acous).transpose(1,0)
        acous_a = acous[:,None,:,:].expand(-1,a.shape[1],-1,-1)
        acous_a = acous_a.reshape(-1,*acous_a.shape[2:])

        acous_i = acous[:,None,:,:].expand(-1,i.shape[1],-1,-1)
        acous_i = acous_i.reshape(-1,*acous_i.shape[2:])
        a = torch.tensor(a)
        i = torch.tensor(i)
        a = a.reshape(-1,*a.shape[2:])
        i = i.reshape(-1,*i.shape[2:])

        index = np.arange(acous_a.shape[0])
        np.random.shuffle(index)
        cat_acous = torch.cat((acous_a,acous_i), dim=0)
        cat_ai = torch.cat((a,i),dim=0)

       #initialize variables for lstm  
        self.a_h0 = torch.randn(1,cat_acous.shape[0],self.hidden_size)
        self.a_c0 = torch.randn(1,cat_acous.shape[0],self.hidden_size)
        self.qa_h0 = torch.randn(1,cat_ai.shape[0],self.hidden_qas_size)
        self.qa_c0 = torch.randn(1,cat_ai.shape[0],self.hidden_qas_size)
        _,(ahn,acn) = self.lstm(cat_acous,(self.a_h0,self.a_c0))
        _,(cat_ai_hn,cat_ai_cn) = self.lstm_a(cat_ai,(self.qa_h0,self.qa_c0))
        ahn = ahn.squeeze()
        cat_ai_hn = cat_ai_hn.squeeze()
        acous_a_doc = self.dock1(ahn)
        
        print(cat_ai_hn.shape)
        print(ahn.shape)
        print("concatenated sample shape")
        print(cat_acous.shape)
        print(cat_ai.shape)

        #print(index)
        
        print(a.shape, acous_a.shape)

    
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
    path = "/home/gaoruohan19/project/Social-IQ-new/data/qai_updated/"

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
                if(array.shape[0]!=18): print(folder)
    
    a_arr = np.stack(a_arr,axis=0)
    i_arr = np.stack(i_arr,axis=0)
    return a_arr,i_arr
def calc_F1(a_res,i_res):
    a_pred = torch.argmax(a_res,dim=1)
    i_pred = torch.argmax(i_res,dim=1)
    true = torch.ones(a_res.shape[0],dtype=torch.long)
    false = torch.zeros(i_res.shape[0],dtype=torch.long)
    pred = torch.cat((a_pred,i_pred),dim=0).numpy()
    ground_truth = torch.cat((true,false),dim=0).numpy()
    f1 = f1_score(ground_truth, pred)
    accu = accuracy_score(ground_truth, pred)
    return f1,accu

if __name__=="__main__":

        #if you have enough RAM, specify this as True - speeds things up ;)
        preload=False
        bs=16
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
        ## input dims 
        acous_input_dim = 74
        vis_input_dim = 2208
        trs_input_dim = 768
        input_qas_dim = 768

        ## ffw arch 
        acous_arch = [128,256,128,64]
        vis_arch = [256,128,128,64]
        trs_arch = [256,128,128,64]
        qas_arch = [512,256,128,256]
        judge_arch = [512,256,64,2]

        fuse_dim = 768
        model = classifier(temp_dim,acous_input_dim, vis_input_dim,trs_input_dim, acous_arch, vis_arch, trs_arch, input_qas_dim, qas_arch, fuse_dim, judge_arch)
        optimizer = torch.optim.Adam(model.parameters(),weight_decay = 0.0, lr=0.0001,betas=(0.9,0.999))
        loss = nn.NLLLoss()
        
        print_model(model)
        start = 0
        num_epoch = 200 
        ### load saved model and resume training 
        weights_path = "./fusion_model_weights"
        onlyfiles = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(weights_path) if os.path.isfile(os.path.join(weights_path, f))]
        print(onlyfiles)
        if(len(onlyfiles)!=0):
            model.load_state_dict(torch.load(weights_path+"/model_{}.pth".format(max(onlyfiles))))
            start = max(onlyfiles)
        loss_array = []
        f_array = []
        acc_array = []

        for k in range(start,num_epoch):
            print("starting epoch {k}".format(k=k))
            for j in range(int(ds_size/bs)+1):
                this_trk = trk[j*bs:(j+1)*bs]
                preloaded_train = process_data(this_trk)
                qas,vis,trs,acc = preloaded_train[0],preloaded_train[1],preloaded_train[2],preloaded_train[3]
                a_arr, i_arr = load_qai(this_trk)
                is_nan = contains_nan(acc)
                print("there is nan in acoustic {}".format(is_nan))
                acc = convert_nans(acc)
                a_res, i_res = model(acc,vis,trs,a_arr, i_arr)
                true = torch.ones(a_res.shape[0],dtype=torch.long)
                false = torch.zeros(i_res.shape[0],dtype=torch.long)
                loss_a = loss(a_res, true)
                loss_i = loss(i_res, false)
                loss_tot = loss_a+loss_i
    
                print("total loss is {loss_tot},loss_a is {loss_a},loss_i is {loss_i}".format(loss_tot=loss_tot,loss_a=loss_a,loss_i=loss_i))
    
                loss_tot.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                with torch.no_grad():
                    print("f1 score and accuracy are {f1}, and {accu}".format(f1 = calc_F1(a_res,i_res)[0], accu = calc_F1(a_res,i_res)[1]))
                            
                print("________")
                # break
            if(k % 1 ==0):
                print("saving model_{}".format(k))
                torch.save(model.state_dict(),weights_path+"/model_{}.pth".format(k))
