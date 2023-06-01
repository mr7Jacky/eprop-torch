# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2020-2022 University of Zurich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "models.py" - Spiking RNN network model embedding hardcoded e-prop training procedures.
 
 Project: PyTorch e-prop

 Author:  C. Frenkel, Institute of Neuroinformatics, University of Zurich and ETH Zurich

 Cite this code: BibTeX/APA citation formats auto-converted from the CITATION.cff file in the repository are available 
       through the "Cite this repository" link in the root GitHub repo https://github.com/ChFrenkel/eprop-PyTorch/
       
------------------------------------------------------------------------------
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt


class SRNN(nn.Module):
    
    def __init__(self, n_in, n_rec, n_out, n_t, thr, tau_m, tau_o, b_o, gamma, dt, model, classif, w_init_gain, lr_layer, t_crop, visualize, visualize_light, device):    
        
        super(SRNN, self).__init__()
        self.n_in     = n_in
        self.n_rec    = n_rec
        self.n_out    = n_out
        self.n_t      = n_t
        self.thr      = thr
        self.dt       = dt
        self.alpha    = np.exp(-dt/tau_m)
        self.kappa    = np.exp(-dt/tau_o)
        self.gamma    = gamma
        self.b_o      = b_o
        self.model    = model
        self.classif  = classif
        self.lr_layer = lr_layer
        self.t_crop   = t_crop  
        self.visu     = visualize
        self.visu_l   = visualize_light
        self.device   = device
        
        #Parameters
        self.w_in  = nn.Parameter(torch.Tensor(n_rec, n_in ))
        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec))
        # self.reg_term = torch.zeros(self.n_rec).to(self.device)
        # self.B_out = torch.Tensor(n_out, n_rec).to(self.device)
        self.reset_parameters(w_init_gain)
        
        #Visualization
        if self.visu:
            plt.ion()
            self.fig, self.ax_list = plt.subplots(2+self.n_out+5, sharex=True)
        self.count = 0
        self.winm, self.wrecm,self.wom  = 0, 0, 0
    def reset_parameters(self, gain):
        
        torch.nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0]*self.w_in.data
        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1]*self.w_rec.data
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2]*self.w_out.data
        
    def init_net(self, n_b, n_t, n_in, n_rec, n_out):
        
        #Hidden state
        self.v  = torch.zeros(n_t,n_b,n_rec).to(self.device)
        self.vo = torch.zeros(n_t,n_b,n_out).to(self.device)
        #Visible state
        self.z  = torch.zeros(n_t,n_b,n_rec).to(self.device)
        #Weight gradients
        self.w_in.grad  = torch.zeros_like(self.w_in)
        self.w_rec.grad = torch.zeros_like(self.w_rec)
        self.w_out.grad = torch.zeros_like(self.w_out)
    
    def forward(self, x, yt, do_training):
        
        self.n_b = x.shape[1]    # Extracting batch size
        self.n_t = x.shape[0]
        self.init_net(self.n_b, self.n_t, self.n_in, self.n_rec, self.n_out)    # Network reset
         
        self.w_rec *= (1 - torch.eye(self.n_rec, self.n_rec, device=self.device))         # Making sure recurrent self excitation/inhibition is cancelled
        
        for t in range(self.n_t-1):     # Computing the network state and outputs for the whole sample duration
        
            # Forward pass - Hidden state:  v: recurrent layer membrane potential
            #                Visible state: z: recurrent layer spike output, vo: output layer membrane potential (yo incl. activation function)
            self.v[t+1]  = (self.alpha * self.v[t] + torch.mm(self.z[t], self.w_rec.t()) + torch.mm(x[t], self.w_in.t())) - self.z[t]*self.thr
            self.z[t+1]  = (self.v[t+1] > self.thr).float()
            self.vo[t+1] = self.kappa * self.vo[t] + torch.mm(self.z[t+1], self.w_out.t()) + self.b_o

        if self.classif:        #Apply a softmax function for classification problems
            yo = F.softmax(self.vo,dim=2)
        else:
            yo = self.vo
        if do_training:
            # print(f'Fire Rate:{self.z.sum()/torch.ones_like(self.z).sum()}')
            self.count += 1
            self.grads_batch(x, yo, yt)
        return yo
    
    def grads_batch(self, x, yo, yt):

        self.n_b = x.shape[1]    # Extracting batch size
        self.n_t = x.shape[0]
        # Learning signals
        err = yo - yt
        L = torch.einsum('tbo,or->brt', err, self.w_out)
        # Surrogate derivatives
        h = self.gamma*torch.max(torch.zeros_like(self.v), 1-torch.abs((self.v-self.thr)/self.thr))
   
        # Input and recurrent eligibility vectors for the 'LIF' model (vectorized computation, model-dependent)
        assert self.model == "LIF", "Nice try, but model " + self.model + " is not supported. ;-)"
        alpha_conv  = torch.tensor([self.alpha ** (self.n_t-i-1) for i in range(self.n_t)]).float().view(1,1,-1).to(self.device)
        kappa_conv = torch.tensor([self.kappa ** (self.n_t-i-1) for i in range(self.n_t)]).float().view(1,1,-1).to(self.device)
        trace_in    = F.conv1d(     x.permute(1,2,0), alpha_conv.expand(self.n_in ,-1,-1), padding=self.n_t, groups=self.n_in )[:,:,1:self.n_t+1].unsqueeze(1).expand(-1,self.n_rec,-1,-1)  #n_b, n_rec, n_in , n_t 
        trace_in    = torch.einsum('tbr,brit->brit', h, trace_in )                                                                                                                          #n_b, n_rec, n_in , n_t
        # Eligibility traces
        trace_in     = F.conv1d(   trace_in.reshape(self.n_b,self.n_in *self.n_rec,self.n_t), kappa_conv.expand(self.n_in *self.n_rec,-1,-1), padding=self.n_t, groups=self.n_in *self.n_rec)[:,:,1:self.n_t+1].reshape(self.n_b,self.n_rec,self.n_in ,self.n_t)   #n_b, n_rec, n_in , n_t  
        # Weight gradient updates
        # print(trace_in.shape, L.unsqueeze(2).expand(-1,-1,self.n_in ,-1).shape)
        self.w_in.grad  += self.lr_layer[0]*torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_in ,-1) * trace_in , dim=(0,3)) 
        del trace_in
        
        trace_rec   = F.conv1d(self.z.permute(1,2,0), alpha_conv.expand(self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec)[:,:, :self.n_t  ].unsqueeze(1).expand(-1,self.n_rec,-1,-1)  #n_b, n_rec, n_rec, n_t
        trace_rec   = torch.einsum('tbr,brit->brit', h, trace_rec)                                                                                                                          #n_b, n_rec, n_rec, n_t    
        # Eligibility traces
        trace_rec    = F.conv1d(  trace_rec.reshape(self.n_b,self.n_rec*self.n_rec,self.n_t), kappa_conv.expand(self.n_rec*self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec*self.n_rec)[:,:,1:self.n_t+1].reshape(self.n_b,self.n_rec,self.n_rec,self.n_t)   #n_b, n_rec, n_rec, n_t
        # Weight gradient updates
        self.w_rec.grad += self.lr_layer[1]*torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_rec,-1) * trace_rec, dim=(0,3))
        del trace_rec, alpha_conv, h, L
        
        # Output eligibility vector (vectorized computation, model-dependent)
        
        trace_out  = F.conv1d(self.z.permute(1,2,0), kappa_conv.expand(self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec)[:,:,1:self.n_t+1]  #n_b, n_rec, n_t
        # Weight gradient updates
        self.w_out.grad += self.lr_layer[2]*torch.einsum('tbo,brt->or', err, trace_out)
        
        del trace_out, kappa_conv
        
        # self.winm = max(self.winm, self.w_in.grad.max())
        # self.wrecm = max(self.wrecm, self.w_rec.grad.max())
        # self.wom = max(self.wom, self.w_out.grad.max())
        # if self.count % 100 == 0:
        #     print(self.winm, self.wrecm, self.wom)
        #     self.winm, self.wrecm,self.wom  = 0, 0, 0
        
    def __repr__(self):
        
        return self.__class__.__name__ + ' (' \
            + str(self.n_in) + ' -> ' \
            + str(self.n_rec) + ' -> ' \
            + str(self.n_out) + ') '
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        