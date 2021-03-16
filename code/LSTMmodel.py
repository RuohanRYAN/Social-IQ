import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from model import mylstm
import numpy as np

class DualLstm(nn.Module):
    def __init__(self):
        super(DualLstm, self).__init__()
        self.vlstm = mylstm.MyLSTM(2208, 256, 2)
        self.qlstm = mylstm.MyLSTM(768, 256, 2)
        self.alstm = mylstm.MyLSTM(768, 256, 2)


    def forward(self,q,a,vis):
        q = np.nan_to_num(q)
        a = np.nan_to_num(a)
        vis = np.nan_to_num(vis)

        vout, vhcn = self.vlstm.step(Variable(torch.tensor(vis)))
        qout, qhcn = self.qlstm.step(Variable(torch.tensor(q)), vhcn[0],vhcn[1])
        aout, ahcn = self.alstm.step(Variable(torch.tensor(a)), qhcn[0],qhcn[1])

        return aout, ahcn
