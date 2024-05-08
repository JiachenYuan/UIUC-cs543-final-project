#%%

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Subset
from torch import cat, stack
data_ROOT = 'data'
def init(root):
    global data_ROOT
    data_ROOT = root

def head(data,n):
    return Subset(data,range(n)) 

# class Dataset:
#     def __init__(self,root):
#         self.root = root
#         self.prepare(root)
#     def get_trainset(self,**kargs):
#         return None
#     def get_testset(self,**kargs):
#         return None
#     def get_train_test(self,**kargs):
#         return self.get_trainset(**kargs),self.get_testset(**kargs)
#     def prepare(self,root):
#         pass
        


# class cifer10(Dataset):
#     def get_trainset(self,normalize=False):
#         trans = [
#             transforms.Resize(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),]
#         if normalize:
#             trans += [
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ]
#         transform = transforms.Compose(trans)
#         return datasets.CIFAR10(root=self.root, train=True, 
#                                 download=True, transform=transform)
        

#     def get_testset(self,normalize=False):
#         trans = [
#             transforms.Resize(224),
#             transforms.ToTensor(),]
#         if normalize:
#             trans += [
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ]
#         transform = transforms.Compose(trans)

#         return datasets.CIFAR10(root=self.root, train=False,
#                                 download=True, transform=transform)

# class kitti(Dataset):
#     def get_trainset(self,normalize=False):
#         trans = [
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),]
#         if normalize:
#             trans += [
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ]
#         transform = transforms.Compose(trans)
#         return datasets.Kitti(root=self.root, train=True, 
#                                 download=True, transform=transform)

#     def get_testset(self,normalize=False):
#         trans = [
#             transforms.ToTensor(),]
#         if normalize:
#             trans += [
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ]
#         transform = transforms.Compose(trans)

#         return datasets.Kitti(root=self.root, train=False,
#                                 download=True, transform=transform)
import random
import importlib
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
DATASETS = ['cifar10','ade20k','nyudepthv2',]

# class DepthBaseDataset(Dataset):
#     def __init__(self, crop_size, transform =None):

#         self.count = 0
        
#         basic_transform = [
#             A.HorizontalFlip(),
#             A.RandomCrop(crop_size[0], crop_size[1]),
#             A.RandomBrightnessContrast(),
#             A.RandomGamma(),
#             A.HueSaturationValue()
#         ]
#         self.basic_transform = basic_transform
#         self.to_tensor = transforms.ToTensor()

#     def readTXT(self, txt_path):
#         with open(txt_path, 'r') as f:
#             listInTXT = [line.strip() for line in f]

#         return listInTXT

#     def augment_training_data(self, image, depth):
#         H, W, C = image.shape

#         if self.count % 4 == 0:
#             alpha = random.random()
#             beta = random.random()
#             p = 0.75

#             l = int(alpha * W)
#             w = int(max((W - alpha * W) * beta * p, 1))

#             image[:, l:l+w, 0] = depth[:, l:l+w]
#             image[:, l:l+w, 1] = depth[:, l:l+w]
#             image[:, l:l+w, 2] = depth[:, l:l+w]

#         additional_targets = {'depth': 'mask'}
#         aug = A.Compose(transforms=self.basic_transform,
#                         additional_targets=additional_targets)
#         augmented = aug(image=image, depth=depth)
#         image = augmented['image']
#         depth = augmented['depth']

#         image = self.to_tensor(image)
#         depth = self.to_tensor(depth).squeeze()

#         self.count += 1

#         return image, depth

#     def augment_test_data(self, image, depth):
#         image = self.to_tensor(image)
#         depth = self.to_tensor(depth).squeeze()

#         return image, depth


# class Nyudepthv2(DepthBaseDataset):

#     def prepare(self,force):
#         # import urllib.request
#         dir_data = self.data_path
#         output_splits_dir = os.path.join(dir_data, 'official_splits')
#         if os.path.isdir(output_splits_dir) and not force: 
#             return
#         import gdown

#         # self.dir_data = dir_data
#         # Download the file using urllib
#         filename = 'nyu_depth_v2_labeled.mat'
#         os.makedirs(dir_data,exist_ok=True)
#         # urllib.request.urlretrieve(url, os.path.join(dir_data,filename))
#         file_path = os.path.join(dir_data,filename)
#         if not os.path.isfile(file_path) or force:
#             url = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
#             gdown.download(url, output = file_path, quiet=False)
#         else:
#             print('mat exist')
#         # Extract official train-test splits using custom script
#         from extract_official_train_test_set_from_mat import extract_official_splits

#         # Provide the necessary arguments and run the extraction script
#         input_mat_file = os.path.join(dir_data, 'nyu_depth_v2_labeled.mat')
#         splits_file = os.path.join(dir_data,'splits.mat')
#         extract_official_splits(input_mat_file, splits_file, output_splits_dir)

#     def __init__(self, root, filenames_path='./code/dataset/filenames/', 
#                  train=True, crop_size=(448, 576), scale_size=None,
#                  download=True,transform=None,force=False,verbose=True):
#         super(Nyudepthv2,self).__init__(crop_size)

#         self.scale_size = scale_size

#         self.is_train = train
#         self.data_path = os.path.join(root, 'nyu_depth_v2')
#         if download: self.prepare(force)
#         self.image_path_list = []
#         self.depth_path_list = []

#         if train:
#             self.data_path = self.data_path + '/official_splits/train'
#         else:
#             self.data_path = self.data_path + '/official_splits/test'

#         txt_path = self.data_path + '/filenames.txt'
#         self.filenames_list = self.readTXT(txt_path)

#         phase = 'train' if train else 'test'
#         if verbose:
#             print("Dataset: NYU Depth V2")
#             print("# of %s images: %d" % (phase, len(self.filenames_list)))
#             print('data path:',self.data_path)
#             # print(self.filenames_list)

#     def __len__(self):
#         return len(self.filenames_list)

#     def __getitem__(self, idx):
#         img_path = self.data_path + '/' + self.filenames_list[idx].split(' ')[0]
#         gt_path = self.data_path + '/' + self.filenames_list[idx].split(' ')[1]
#         filename = self.filenames_list[idx].split(' ')[0].replace('/','_')
#         image = cv2.imread(img_path)
#         assert image is not None, f"bad: {img_path}"
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

#         if self.scale_size:
#             image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
#             depth = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))

#         if self.is_train:
#             image, depth = self.augment_training_data(image, depth)
#         else:
#             image, depth = self.augment_test_data(image, depth)

#         depth = depth / 1000.0  # convert in meters
#         return image, depth
#         # return {'image': image, 'depth': depth, 'filename': filename}
 
def crop_dict(x):
    crop = (224,224)
    return {k:transforms.CenterCrop(crop)(v) for k,v in x.items()}

def To_Tensor_dict(x):
    return {k:transforms.ToTensor()(v) for k,v in x.items()}
from tqdm import tqdm
def get_set(name,train=True,crop=(224,224),cache_dir='data',download=True):
    ret = None
    if name in ['cifer10','cifar10']: #classification
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),])
        ret = datasets.CIFAR10(cache_dir,train,transform,download=download)
    elif name == 'kitti': #detection
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),])
        ret = datasets.Kitti(cache_dir,train,transform,download=download)
    elif name == 'ade20k': #segmentation
        from datasets import load_dataset
        key = "scene_parse_150"
        if train:
            ret = load_dataset(key,split='train',data_dir = cache_dir, cache_dir=cache_dir)
        else:
            ret = load_dataset(key,split='validation',data_dir = cache_dir, cache_dir=cache_dir)
        ret = ret.map(crop_dict)
        ret = ret.map(To_Tensor_dict)
        ret = ret.take(100)
        ret = map(dict.values,ret)
        ret = map(list,ret)
        ret = list(ret)
        # transform = transforms.Compose([
        #     transforms.CenterCrop(crop),])
        # ret.set_transform(transform)
    elif name == 'nyudepthv2': #depth
        from torchvision.transforms import Compose,RandomRotation,CenterCrop,ToTensor
        from datasets import load_dataset

        key = "sayakpaul/nyu_depth_v2"
        if train:
            ret = load_dataset(key,split='train',data_dir=cache_dir,cache_dir=cache_dir)#,streaming=True)
            n=47584
            # n = len(ret)
            trans = Compose([
                # RandomRotation(degrees=(0, 90)),
                
                CenterCrop(crop),
                ToTensor(),
                ])
            def transform(e):
                angle = random.randint(0, 90)
                # e["image"] = [trans(i.rotate(angle)) for i in e["image"]]
                # e['depth_map'] = [trans(i.rotate(angle)) for i in e['depth_map']]
                e["image"] = [trans(i) for i in e["image"]]
                e['depth_map'] = [trans(i) for i in e['depth_map']]
                return e
        else:
            ret = load_dataset(key,split='validation',data_dir=cache_dir,cache_dir=cache_dir)#,streaming=True)
            n=654
            n = 200
            trans = Compose([
                # RandomRotation(degrees=(0, 90)),
                CenterCrop(crop),
                ToTensor(),
                ])
            def transform(e):
                e["image"] = [trans(i) for i in e["image"]]
                e['depth_map'] = [trans(i) for i in e['depth_map']]
                return e



        
        ret.set_transform(transform)
        # ret = ret.map(crop_dict)
        # ret = ret.map(To_Tensor_dict)
        # ret = ret.take(n)
        # ret = ret.with_format("torch")
        ret = head(ret,n)
        l = n
        # ret = map(dict.values,ret)
        # ret = map(list,ret)
        # ret = list(tqdm(ret,total = n))

        # transform = transforms.Compose([
        #     transforms.CenterCrop(crop),])
        # ret.set_transform(transform)
        
    elif name == 'nyudepthv2s': #depth
        from dataloaders import nyu_dataloader
    
        if train:
            train_set = nyu_dataloader.NYUDataset('data/official_splits/train', type='train')
            return train_set
        else:
            val_set = nyu_dataloader.NYUDataset('data/official_splits/val', type='val')
            return val_set
    # elif name == 'addmore':
    #     ret = 
    return ret,l


def get_test_piece(name,root='data',n=10,backbone=None,):
    from torch.utils.data import DataLoader
    dataset = get_set(name, train=True, download=True,cache_dir=root)
    dataset = head(dataset,n)
    loader = DataLoader(dataset, batch_size=5, 
                            shuffle=True, num_workers=0)
    piece = next(iter(loader))
    return piece

    
#%% 
if __name__ == '__main__':
    from utils import silence
    from models import BACKBONES

    print('search errors...')
    for name in DATASETS:
        print(name,': ',end='') 
        try:
            kargs = {}
            if name in ['nyudepthv2']: kargs = {'verbose':False}
            with silence():
                trainset = get_set(name,root='data', train=True, download=True,**kargs)
                testset = get_set(name,root='data', train=False, download=True,**kargs)
            if trainset == None:
                print('bad train',end='')
            elif testset == None:
                print('bad test',end='')
            else:
                print('Good',end='')
        except:
            print( 'Error',end='')
        print()
                
    print('peek data...')
    for name in DATASETS:
            print(name,': ') 
            try:
                with silence():
                    ret = get_test_piece(name)
            except:
                ret = None
                print('Error')
            finally:
                if ret == None: continue
                for i in range(len(ret)) if type(ret)==list else ret.keys():
                    try:
                        print(i,ret[i].shape)
                    except:
                        print(i,len(i))
            

def show_shape(name,root='data',n=10):
    ret = get_test_piece(name,root,n)

    for i in range(len(ret)) if type(ret)==list else ret.keys():
        try:
            print(i,ret[i].shape)
        except:
            print(i,len(i))

# %%
