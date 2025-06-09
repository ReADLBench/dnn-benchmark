import torch
from gpu.image_classification.CIFAR10.fp32.resnet import resnet_cifar10
from gpu.image_classification.CIFAR10.fp32.mobilenet import mobilenetv2_cifar10
from gpu.image_classification.CIFAR10.fp32.densenet import densenet_cifar10
from gpu.image_classification.CIFAR10.fp32.vgg import vgg_cifar10
from gpu.image_classification.CIFAR10.fp32.googlenet import googlenet_cifar10

from gpu.image_classification.CIFAR100.fp32.resnet import resnet_cifar100
from gpu.image_classification.CIFAR100.fp32.densenet import densenet_cifar100
from gpu.image_classification.CIFAR100.fp32.googlenet import googlenet_cifar100

from gpu.image_classification.GTSRB.fp32.resnet import resnet_GTSRB
from gpu.image_classification.GTSRB.fp32.vgg import vgg_GTSRB
from gpu.image_classification.GTSRB.fp32.densenet import densenet_GTSRB

from gpu.image_segmentation.PASCAL_VOC.fp32.DeepLabV3.DeepLabV3 import deeplabv3_resnet50

# --- Available Datasets, and Models ---

DATASETS = ['CIFAR10', 'CIFAR100', 'GTSRB', 'PASCAL_VOC']

MODEL_DATASET_COMPATIBILITY = {
    'CIFAR10': ['ResNet20', 'ResNet32', 'ResNet44', 'DenseNet121', 'DenseNet161',
                'GoogLeNet', 'MobileNetV2', 'Vgg11_bn', 'Vgg13_bn'],
    'CIFAR100': ['ResNet18', 'GoogLeNet', 'DenseNet121'],
    'GTSRB': ['ResNet20', 'Vgg11_bn', 'DenseNet121'],
    'PASCAL_VOC': ['deeplabv3_resnet50, DeepLabV3_resnet50_state_dict'],
}


NETWORK_NAME = 'ResNet20'
DATASET_NAME = 'CIFAR10'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_from_dict(network, device, path, function=None):
    if '.th' in path:
        state_dict = torch.load(path, map_location=device)['state_dict']
        print('state_dict loaded')
    else:
        state_dict = torch.load(path, map_location=device)
        print('state_dict loaded')

    if function is None:
        clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    else:
        clean_state_dict = {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}

    network.load_state_dict(clean_state_dict, strict=False)
    print('state_dict loaded into network')
    

if DATASET_NAME == 'PASCAL_VOC' or 'COCOdetection':
        print(f'Loading network {NETWORK_NAME} ...')
        if 'DeepLabV3_resnet50' in NETWORK_NAME:
            network_path = '/dnn-benchmark/pytorch/gpu/image_segmentation/PASCAL_VOC/fp32/Deeplabv3/deeplabv3_resnet50/PyTorch/deeplabv3_resnet50.pth'
            network = torch.load(network_path, map_location=device)
        elif 'DeepLabV3_resnet50_state_dict' in NETWORK_NAME:
            model = deeplabv3_resnet50(weights=None, num_classes=21, aux_loss=True)
            
            state_dict_path = '/dnn-benchmark/pytorch/gpu/image_segmentation/PASCAL_VOC/fp32/Deeplabv3/deeplabv3_resnet50/PyTorch/deeplabv3_resnet50.pt'
            state_dict = torch.load(state_dict_path, map_location='cpu')
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # solo se necessario
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        else:
            raise ValueError(f'Invalid network name {NETWORK_NAME}')
    
if DATASET_NAME == 'CIFAR10':
    print(f'Loading network {NETWORK_NAME} ...')   
    if 'ResNet20' in NETWORK_NAME:
        network = resnet_cifar10.resnet20() 
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR10/fp32/resnet/ResNet20/PyTorch/ResNet20.th'
    elif 'ResNet32' in NETWORK_NAME:
        network = resnet_cifar10.resnet32()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR10/fp32/resnet/ResNet32/PyTorch/ResNet32.th'
    elif 'ResNet44' in NETWORK_NAME:
        network = resnet_cifar10.resnet44()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR10/fp32/resnet/ResNet44/PyTorch/ResNet44.th'
    elif 'DenseNet121' in NETWORK_NAME:
        network = densenet_cifar10.densenet121()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR10/fp32/densenet/DenseNet121/PyTorch/DenseNet121.th'
    elif 'DenseNet161' in NETWORK_NAME:
        network = densenet_cifar10.densenet161()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR10/fp32/densenet/DenseNet161/PyTorch/DenseNet161.th'
    elif 'GoogLeNet' in NETWORK_NAME:
        network = googlenet_cifar10.googlenet()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR10/fp32/googlenet/GoogLeNet/PyTorch/GoogLeNet.th'
    elif 'Vgg11_bn' in NETWORK_NAME:
        network = vgg_cifar10.vgg11_bn()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR10/fp32/vgg/Vgg11_bn/PyTorch/Vgg11_bn.th'
    elif 'Vgg13_bn' in NETWORK_NAME:
        network = vgg_cifar10.vgg13_bn()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR10/fp32/vgg/Vgg13_bn/PyTorch/Vgg13_bn.th'
    elif 'MobileNetV2' in NETWORK_NAME:
        network = mobilenetv2_cifar10.MobileNetV2() 
        
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR10/fp32/mobilenet/MobileNetV2/PyTorch/MobileNetV2.th'
        # NOTE: This checkpoint uses "net" as the key for the state dictionary,
        # instead of the more common "state_dict". Make sure to access it accordingly.
        state_dict = torch.load(weight_path, map_location=device)["net"]
        function = None
        if function is None:
            clean_state_dict = {
                key.replace("module.", ""): value for key, value in state_dict.items()
            }
        else:
            clean_state_dict = {
                key.replace("module.", ""): function(value)
                if not (("bn" in key) and ("weight" in key))
                else value
                for key, value in state_dict.items()
            }
        network.load_state_dict(clean_state_dict, strict=False)
    else:
        raise ValueError(f'Invalid network name {network}')

    # Load the weights
    if 'MobileNetV2' not in NETWORK_NAME:
        if 'ResNet' in NETWORK_NAME:
            network_path = weight_path
        else:
            network_path = weight_path
    
        load_from_dict(network=network,
                        device=device,
                        path=network_path)
    
elif DATASET_NAME == 'CIFAR100':
    print(f'Loading network {NETWORK_NAME} ...')
    if 'ResNet18' in NETWORK_NAME:
        network = resnet_cifar100.resnet18()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR100/fp32/resnet/ResNet18/PyTorch/ResNet18.th'
    elif 'DesneNet121' in NETWORK_NAME:
        network = densenet_cifar100.densenet121()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR100/fp32/densenet/DenseNet121/PyTorch/DenseNet121.th'
    elif 'GoogLeNet' in NETWORK_NAME:
        network = googlenet_cifar100.googlenet()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/CIFAR100/fp32/googlenet/GoogLeNet/PyTorch/GoogLeNet.th'
    else:
        raise ValueError(f'Invalid network name {network}')
    
    # Load the weights
    network_path = weight_path
    function = None
    
    state_dict = torch.load(network_path, map_location=device)['state_dict'] if '.th' in network_path else torch.load(network_path, map_location=device)
    clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()} if function is None else {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}
    network.load_state_dict(clean_state_dict, strict=False)

elif DATASET_NAME == 'GTSRB':
    print(f'Loading network {NETWORK_NAME} ...')
    if 'ResNet20' in NETWORK_NAME:
        network = resnet_GTSRB.resnet20()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/GTSRB/fp32/resnet/ResNet20/PyTorch/ResNet20.th'
    elif 'DenseNet121' in NETWORK_NAME:
        network = densenet_GTSRB.densenet121()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/GTSRB/fp32/densenet/DenseNet121/PyTorch/DenseNet121.th'
    elif 'Vgg11_bn' in NETWORK_NAME:
        network = vgg_GTSRB.vgg11_bn()
        weight_path = '/dnn-benchmark/pytorch/gpu/image_classification/GTSRB/fp32/vgg/Vgg11_bn/PyTorch/Vgg11_bn.th'
    else:
        raise ValueError(f'Invalid network name {network}')
    
    network_path = weight_path
    
    load_from_dict(network=network,
                    device=device,
                    path=network_path)

network.to(device)
network.eval()
    
    
