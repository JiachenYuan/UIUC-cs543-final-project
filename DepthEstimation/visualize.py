#%%
from more_itertools import callback_iter
import models as MODELS
import data as DATA
import save as SAVE


import numpy as np
import matplotlib.pyplot as plt

def vis_tensor(t):
    try:
        if t.device.type == 'cuda':
            t=t.cpu()
    except:
        pass
    # if len(t.shape) == 4:
    t = t.squeeze()
    t = t.detach().numpy()

    # num_channels = t.shape[0]
    # if num_channels == 1:
    #     t = np.repeat(t, 3, axis=0)
    if len(t.shape) == 3:
        t = t.transpose(1, 2, 0)
    print(t.shape)

    if (t!=0).any():
        t = t / np.max(np.abs(t))

    plt.imshow(t)
    plt.axis('off')
    plt.show()

#%% 
def get_trained(modeln='resnet',task='depth',datan='nyudepthv2',mode='e2e',new='back'):
    SAVE.init('output',modeln,task,mode,datan,test=False,new=new)
    model, call = MODELS.get_model(modeln,task,mode)
    model.load_state_dict(SAVE.load_model())
    return model,call

#%%
def call_trained(modeln='resnet',task='depth',datan='nyudepthv2',mode='e2e',new='back',train=False):
    device = 'cuda:0'
    model, call = get_trained(modeln,task,datan,mode,new)
    model.eval()
    data = DATA.get_set(datan,train=train)[0]
    index = 0
    if task == 'depth':
        input = data[index]['image']
        gt = data[index]['depth_map']
        if modeln in ['resnet','resnet50']:
            out = call(model.to(device),input[None,:,:,:].to(device))[0]
            return input,gt,out

        if modeln == 'mae':
            out = call(model,input[None,:,:,:])[0]
            return input,gt,out


#%%
import torch
from metrics import Result, AverageMeter
from tqdm import tqdm
from torch import nn
def evaluate_model(model,train,loader,device='cuda:0',callback=None):
    class MaskedL1Loss(nn.Module):
        def __init__(self):
            super(MaskedL1Loss, self).__init__()

        def forward(self, pred, target):
            assert pred.dim() == target.dim(), "inconsistent dimensions"
            valid_mask = (target > 0).detach()
            diff = target - pred
            diff = diff[valid_mask]
            self.loss = diff.abs().mean()
            return self.loss

    criterion = MaskedL1Loss()
    eval_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.values() if isinstance(data,dict) else data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                pred = train(model,inputs)
            loss = criterion(pred,labels).item()
            eval_loss += loss
            result = Result()
            result.evaluate(pred,labels)
            if callback: callback(inputs,pred,labels)
    return eval_loss, result

import time
def timeit(f,n=10):
    start_time = time.time()
    for _ in range(n):
        f()
    end_time = time.time()
    execution_time = (end_time - start_time)/n
    return execution_time

from torch.utils.data import DataLoader
def evalu(modeln,task='depth',datan='nyudepthv2',mode='e2e',new=False,device='cuda:0',callback=None):
    model, call = get_trained(modeln,task,datan,mode,new)
    valset  = DATA.get_set(datan,cache_dir='data', train=False, download=True)[0]
    valloader = DataLoader(valset, batch_size=8,
                                         shuffle=False, num_workers=0)
    total_loss,result=evaluate_model(model.to(device),call,valloader,callback=callback)

    piece = valset[0]['image'][None,:,:,:].to('cuda:0')
    result.gpu_time = timeit(lambda:call(model,piece),10)
    return total_loss,result

from scipy.ndimage import uniform_filter1d
def smooth(c):
    return   uniform_filter1d(c, size=5, mode='nearest')

def draw_loss(train,test,name=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train, label='Train Loss')
    plt.plot(smooth(test), label='Test Loss')
    plt.title('Training and Test Losses' + f' for {name}' if name else '')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def draw_acc(mses,names,met='MSE'):
    colors = ['red','purple','green']
    window_size = 3
    for mse, name, c in zip(mses,names, colors):
        test_correctness_smoothed = smooth(mse)
        plt.plot(test_correctness_smoothed, label=f'{met} {name}', color=c)
    plt.title(f'{met} Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(f'{met} Score')
    plt.legend()
    plt.show()

#%%
def vis_log(modeln='resnet',task='depth',datan='nyudepthv2',mode='e2e',new='back'):
    SAVE.init('output',modeln,task,mode,datan,test=False,new=new)
    lt, lv, mse= SAVE.load_log()

#%% 
if __name__ == "__main__":
    modeln='mae50'
    task='depth'
    datan='nyudepthv2'
    mode='e2e'
    new='berhu'

    x,gt,y = call_trained('mae',datan=datan,new='back',train=False)
    vis_tensor(x)
    vis_tensor(y)
# %%
if __name__ == "__main__":
    modeln='resnet50'
    task='depth'
    datan='nyudepthv2'
    mode='e2e'
    new='berhu'
    # def callback(inputs,pred,labels):
    #     print('x:')
    #     vis_tensor(inputs[0])
    #     print('pred:')
    #     vis_tensor(pred[0])
    #     print('label:')
    #     vis_tensor(labels[0])
    callback = None
    total_loss,result=evalu(modeln,datan=datan,new=new,callback=callback)
    print(
        """
        |rmse|rel|log10|d1|d2|d3|time in gpu|
        | -- | -- | -- | -- | -- | -- |
        |{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.4f}|
        """.format(
            result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                    result.delta3, result.gpu_time))
# %%
if __name__ == "__main__":
    models=['resnet','resnet50','mae']
    names=['DCRN18','DCRN50','Swin-DPT']
    task='depth'
    datan='nyudepthv2'
    mode='e2e'
    new='berhu'

    l=[]
    for model, name in zip(models,names):
        SAVE.init('output',model,task,mode,datan,test=False,new=new)
        llt, llv, lmse= SAVE.load_log()
        llt = llt[2:36]
        llv = llv[2:36]
        lmse = lmse[2:36]
        draw_loss(llt,llv,name=name)
        l.append(lmse)

    draw_acc(l,names,"MSE")
# %%

# %%
# %%
