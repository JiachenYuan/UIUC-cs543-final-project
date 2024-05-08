#%%
from torchvision import models
from transformers import ViTForImageClassification, ResNetForImageClassification, ViTMAEForPreTraining, AutoImageProcessor, AutoFeatureExtractor, ViTMAEConfig, ViTConfig, ViTMAEModel, ViTImageProcessor, TrainingArguments, Trainer, image_processing_utils
from torch import nn
import FCRN

TASKS = ['classification','segmentation','depth']
BACKBONES = ['resnet','mae']

def unfreeze(param):
    param.requires_grad = True

def freeze(param):
    param.requires_grad = False

def get_preprocessor(backbone_type):
    if backbone_type == 'resnet':
        return AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
    if backbone_type == 'mae':
        return ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
        # return ViTImageProcessor.from_pretrained('facebook/vit-mae-base')

def get_model(backbone_type,task,mode ='last',n_classes=10):
    added = None
    model = None
    if task == 'classification':
        if backbone_type == 'resnet':
            model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152")
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
            added = [model.classifier[1]]
        elif backbone_type == 'mae':
            model = ViTForImageClassification.from_pretrained('facebook/vit-mae-base', num_labels=10)
            added = [model.classifier]
    elif task == 'segmentation':
        if backbone_type == 'resnet':
            import transformers
            backbone_config = transformers.ResNetConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
            config = transformers.UperNetConfig(backbone_config=backbone_config)#,use_pretrained_backbone=True)
            model = transformers.UperNetForSemanticSegmentation(config)
            added = nn.ModuleList([model.decode_head,model.auxiliary_head])
        elif backbone_type == 'mae':
            import transformers
            backbone_config = transformers.SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
            config = transformers.UperNetConfig(backbone_config=backbone_config)#,use_pretrained_backbone=True)
            model = transformers.UperNetForSemanticSegmentation(config)
            added = nn.ModuleList([model.decode_head,model.auxiliary_head])
    elif task == 'depth':
        if backbone_type == 'resnet':
            model = FCRN.ResNet(dataset='nyudepthv2',output_size=(224,224),layers=18,pretrained=False)
            added = []
            def train(model,inputs):
            #     size = inputs.shape[2:]
                predited_depth = model(inputs)
            #     prediction = nn.functional.interpolate(
            #         predited_depth.unsqueeze(1),
            #         size=size,
            #         mode="bicubic",
            #         align_corners=False,
            #     )
                # return prediction
                return predited_depth
        elif backbone_type == 'resnet50':
            model = FCRN.ResNet(dataset='nyudepthv2',output_size=(224,224),layers=50,pretrained=False)
            added = []
            def train(model,inputs):
            #     size = inputs.shape[2:]
                predited_depth = model(inputs)
            #     prediction = nn.functional.interpolate(
            #         predited_depth.unsqueeze(1),
            #         size=size,
            #         mode="bicubic",
            #         align_corners=False,
            #     )
                # return prediction
                return predited_depth
        elif backbone_type == 'mae':
            from transformers import Swinv2Config, DPTConfig, AutoModelForDepthEstimation

            backbone_config = Swinv2Config(out_features=["stage1", "stage2", "stage3","stage4"])
            config = DPTConfig(backbone_config=backbone_config,num_hidden_layers=6)
            model = AutoModelForDepthEstimation.from_config(config)
            modelc = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-swinv2-tiny-256")
            model.backbone = modelc.backbone
            added = [model.neck,model.head]
            preprocessor = AutoImageProcessor.from_pretrained("Intel/dpt-swinv2-tiny-256")
            def train(model,inputs):
                size = inputs.shape[2:]
                device = inputs.device
                inputs = preprocessor(inputs, return_tensors="pt").to(device)
                predicted_depth = model(**inputs).predicted_depth
                prediction = nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                )
                # return predicted_depth
                return prediction
                
    elif task == 'detection':
        if backbone_type == 'resnet':
            pass
        elif backbone_type == 'mae':
            pass
    model.added = nn.ModuleList(added)
    if mode == 'last':
        map(freeze,model.parameters())
        map(unfreeze,model.added)
    elif mode == 'e2e':
        map(unfreeze,model.parameters())

    return model,train


#%%
if __name__ == '__main__':

    import contextlib
    import os
    import sys
    @contextlib.contextmanager
    def silence():
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            devnull.close()

    print('test','get_models','...')
    for task in TASKS:
        for backbone in BACKBONES:
            stat = None
            try:
                with silence():
                    model = get_model(backbone,task)
                if model != None:
                    stat = 'Good'
                else:
                    stat = 'None'
            except:
                stat = 'Error'
            print(backbone,task,':',stat)            

#%%
if __name__ == '__main__':
    print('test training','...')
    import data
    # criterion = nn.CrossEntropyLoss()
    # for backbone in BACKBONES:
    #     try:
    #         with silence():
    #             model = get_model(backbone,'classification') 
    #             inputs, labels = data.get_test_piece('cifar10',backbone=backbone)
    #             preprocessor = get_preprocessor(backbone)    
    #             inputs = preprocessor(inputs, return_tensors="pt")
    #             outputs = model(**inputs).logits
    #             loss = criterion(outputs, labels).item()
    #         print('classification',backbone,':',end='')
    #         print('Good')
    #     except:
    #         print('classification',backbone,':',end='')
    #         print('Bad')

    # for backbone in BACKBONES:
    #     try:
    #         with silence():
    #             model = get_model(backbone,'segmentation') 
    #             inputs, labels = data.get_test_piece('ade20k',backbone=backbone)
    #             preprocessor = get_preprocessor(backbone)    
    #             inputs = preprocessor(inputs, return_tensors="pt")
    #             outputs = model(**inputs).logits
    #             loss = criterion(outputs, labels).item()
    #         print('seg',backbone,':',end='')
    #         print('Good')
    #     except:
    #         print('seg',backbone,':',end='')
    #         print('Bad')

    criterion = nn.CrossEntropyLoss()
    for backbone in BACKBONES:
        try:
            with silence():
                model ,train = get_model(backbone,'depth') 
                inputs, labels = data.get_test_piece('nyudepthv2',backbone=backbone)
            loss = train(model,inputs,labels)
            print('depth',backbone,':',end='')
            print('Good')
        except:
            print('depth',backbone,':',end='')
            print('Bad')

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
# %%