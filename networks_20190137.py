import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import h5py
import json
import copy
import time
from datetime import datetime

BOARD_SIZE=8


# -- Loss Function
def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)


class CNN(nn.Module):
    def __init__(self, conf):

        super(CNN, self).__init__()  
        
        # Essential variables
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_CNN/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        
        # ------------------ 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        
        self.fc1 = nn.Linear(256*1*1, 128)
        self.fc2 = nn.Linear(128, self.board_size*self.board_size)
        # ------------------ 
        

    # ------------------ 
    def forward(self, seq):
        x = F.relu(self.conv1(seq))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # Reshape so it fits
        x = x.view(-1, 256*1*1)

        x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out
    # ------------------ 
    
    
    def train_all(self, train, dev, num_epoch, device, optimizer):

        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")

        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()

        for epoch in range(1, num_epoch+1):

            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0

            for batch, labels, _ in tqdm(train):


                # ------------
                # print(f"BATCH_SIZE : {batch.size()}")
                # print(f"LABELS_SIZE : {labels.size()}")
                # ------------


                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()

            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            # Set the module in eval mode
            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
            
            # Set the module in training mode
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')

        # Set the module in eval mode
        self.eval()

        # -- Evaluation after training
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep