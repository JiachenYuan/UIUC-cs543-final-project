import numpy as np
import os
import torch


save_ROOT = 'output'
from datetime import datetime

def init(root,backbone_type,task,mode,data_name,test,new=False):
    global save_ROOT
    subdir = f'{backbone_type}-{task}-{data_name}-{mode}'
    if test:
        subdir += '-codetest'
    if new:
        if isinstance(new,str):
            subdir += '-' + new
        else: 
            now = datetime.now()
            subdir += '-' + now.strftime("%m%d%H%M")
        
    save_ROOT = os.path.join(root,subdir)

    os.makedirs(root,exist_ok=True)
    os.makedirs(save_ROOT,exist_ok=True)

def save_log(train_loss,test_loss,test_acc):
    train_loss, test_loss, test_acc = list(map(np.array,[train_loss,test_loss,test_acc]))
    np.save(os.path.join(save_ROOT,'train_losses.npy'), train_loss)
    np.save(os.path.join(save_ROOT,'test_losses.npy'), test_loss)
    np.save(os.path.join(save_ROOT,'test_correctness.npy'), test_acc)

def load_log():
    if not os.path.exists(os.path.join(save_ROOT,'train_losses.npy')):
        return [],[],[]
    return \
    np.load(os.path.join(save_ROOT,'train_losses.npy')),\
    np.load(os.path.join(save_ROOT,'test_losses.npy')),\
    np.load(os.path.join(save_ROOT,'test_correctness.npy'))

def save_model(model):
    torch.save(model.state_dict(), os.path.join(save_ROOT,'model.pt'))

def load_model():
    return torch.load(os.path.join(save_ROOT,'model.pt'))