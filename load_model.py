import torch
import sys
from pytorch.gpu.image_classification.CIFAR10.fp32.resnet import resnet_cifar10
from pytorch.gpu.image_classification.CIFAR10.fp32.mobilenet import mobilenetv2_cifar10
from pytorch.gpu.image_classification.CIFAR10.fp32.densenet import densenet_cifar10
from pytorch.gpu.image_classification.CIFAR10.fp32.vgg import vgg_cifar10
from pytorch.gpu.image_classification.CIFAR10.fp32.googlenet import googlenet_cifar10

from pytorch.gpu.image_classification.CIFAR100.fp32.resnet import resnet_cifar100
from pytorch.gpu.image_classification.CIFAR100.fp32.densenet import densenet_cifar100
from pytorch.gpu.image_classification.CIFAR100.fp32.googlenet import googlenet_cifar100

from pytorch.gpu.image_classification.GTSRB.fp32.resnet import resnet_GTSRB
from pytorch.gpu.image_classification.GTSRB.fp32.vgg import vgg_GTSRB
from pytorch.gpu.image_classification.GTSRB.fp32.densenet import densenet_GTSRB

from pytorch.gpu.image_segmentation.PASCAL_VOC.fp32.DeepLabV3.DeepLabV3 import deeplabv3_resnet50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ['CIFAR10', 'CIFAR100', 'GTSRB', 'PASCAL_VOC']

MODEL_DATASET_COMPATIBILITY = {
    'CIFAR10': ['ResNet20', 'ResNet32', 'ResNet44', 'DenseNet121', 'DenseNet161',
                'GoogLeNet', 'MobileNetV2', 'Vgg11_bn', 'Vgg13_bn'],
    'CIFAR100': ['ResNet18', 'GoogLeNet', 'DenseNet121'],
    'GTSRB': ['ResNet20', 'Vgg11_bn', 'DenseNet121'],
    'PASCAL_VOC': ['DeepLabV3_resnet50', 'DeepLabV3_resnet50_state_dict'],
}


def menu(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, start=1):
        print(f"[{i}] {opt}")
    choice = input("Select option: ")
    
    if not choice.isdigit() or int(choice) not in range(1, len(options) + 1):
        print("Invalid selection.")
        sys.exit(1)
        
    return options[int(choice) - 1]


def load_from_dict(network, device, path, function=None):
    if '.th' in path:
        state_dict = torch.load(path, map_location=device)['state_dict']
    else:
        state_dict = torch.load(path, map_location=device)

    clean_state_dict = {
        key.replace('module.', ''): function(value) if function and not (('bn' in key) and ('weight' in key)) else value
        for key, value in state_dict.items()
    }
    network.load_state_dict(clean_state_dict, strict=False)


def load_model(dataset, model_name):
    print(f"Loading {model_name} for {dataset}...")
    
    if dataset == 'CIFAR10':
        if model_name == 'ResNet20':
            model = resnet_cifar10.resnet20()
            weight_path = './pytorch/gpu/image_classification/CIFAR10/fp32/resnet/ResNet20/PyTorch/ResNet20.th'
        elif model_name == 'ResNet32':
            model = resnet_cifar10.resnet32()
            weight_path = './pytorch/gpu/image_classification/CIFAR10/fp32/resnet/ResNet32/PyTorch/ResNet32.th'
        elif model_name == 'ResNet44':
            model = resnet_cifar10.resnet44()
            weight_path = './pytorch/gpu/image_classification/CIFAR10/fp32/resnet/ResNet44/PyTorch/ResNet44.th'
        elif model_name == 'DenseNet121':
            model = densenet_cifar10.densenet121()
            weight_path = './pytorch/gpu/image_classification/CIFAR10/fp32/densenet/DenseNet121/PyTorch/DenseNet121.pt'
        elif model_name == 'DenseNet161':
            model = densenet_cifar10.densenet161()
            weight_path = './pytorch/gpu/image_classification/CIFAR10/fp32/densenet/DenseNet161/PyTorch/DenseNet161.pt'
        elif model_name == 'GoogLeNet':
            model = googlenet_cifar10.googlenet()
            weight_path = './pytorch/gpu/image_classification/CIFAR10/fp32/googlenet/GoogLeNet/PyTorch/GoogLeNet.pt'
        elif model_name == 'Vgg11_bn':
            model = vgg_cifar10.vgg11_bn()
            weight_path = './pytorch/gpu/image_classification/CIFAR10/fp32/vgg/Vgg11_bn/PyTorch/Vgg11_bn.pt'
        elif model_name == 'Vgg13_bn':
            model = vgg_cifar10.vgg13_bn()
            weight_path = './pytorch/gpu/image_classification/CIFAR10/fp32/vgg/Vgg13_bn/PyTorch/Vgg13_bn.pt'
        elif model_name == 'MobileNetV2':
            model = mobilenetv2_cifar10.MobileNetV2()
            weight_path = './pytorch/gpu/image_classification/CIFAR10/fp32/mobilenet/MobileNetV2/PyTorch/MobileNetV2.pt'
            # NOTE: Custom "net" key handling
            state_dict = torch.load(weight_path, map_location=device)['net']
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
            return model
        else:
            raise ValueError("Invalid CIFAR10 model.")
        load_from_dict(model, device, weight_path)

    elif dataset == 'CIFAR100':
        if model_name == 'ResNet18':
            model = resnet_cifar100.resnet18()
            weight_path = './pytorch/gpu/image_classification/CIFAR100/fp32/resnet/ResNet18/PyTorch/ResNet18_CIFAR100.pth'
        elif model_name == 'DenseNet121':
            model = densenet_cifar100.densenet121()
            weight_path = './pytorch/gpu/image_classification/CIFAR100/fp32/densenet/DenseNet121/PyTorch/DenseNet121_CIFAR100.pth'
        elif model_name == 'GoogLeNet':
            model = googlenet_cifar100.googlenet()
            weight_path = './pytorch/gpu/image_classification/CIFAR100/fp32/googlenet/GoogLeNet/PyTorch/GoogLeNet_CIFAR100.pth'
        else:
            raise ValueError("Invalid CIFAR100 model.")
        load_from_dict(model, device, weight_path)

    elif dataset == 'GTSRB':
        if model_name == 'ResNet20':
            model = resnet_GTSRB.resnet20()
            weight_path = './pytorch/gpu/image_classification/GTSRB/fp32/resnet/ResNet20/PyTorch/ResNet20_GTSRB.pt'
        elif model_name == 'DenseNet121':
            model = densenet_GTSRB.densenet121()
            weight_path = './pytorch/gpu/image_classification/GTSRB/fp32/densenet/DenseNet121/PyTorch/DenseNet121_GTSRB.pt'
        elif model_name == 'Vgg11_bn':
            model = vgg_GTSRB.vgg11_bn()
            weight_path = './pytorch/gpu/image_classification/GTSRB/fp32/vgg/Vgg11_bn/PyTorch/Vgg11_bn_GTSRB.pt'
        else:
            raise ValueError("Invalid GTSRB model.")
        load_from_dict(model, device, weight_path)

    elif dataset == 'PASCAL_VOC':
        if model_name == 'DeepLabV3_resnet50':
            weight_path = './pytorch/gpu/image_segmentation/PASCAL_VOC/fp32/Deeplabv3/deeplabv3_resnet50/PyTorch/deeplabv3_resnet50.pth'
            model = torch.load(weight_path, map_location=device, weights_only =False)
        elif model_name == 'DeepLabV3_resnet50_state_dict':
            model = deeplabv3_resnet50(weights=None, num_classes=21, aux_loss=True)
            weight_path = './pytorch/gpu/image_segmentation/PASCAL_VOC/fp32/Deeplabv3/deeplabv3_resnet50/PyTorch/deeplabv3_resnet50.pt'
            state_dict = torch.load(weight_path, map_location=device)
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
        else:
            raise ValueError("Invalid PASCAL VOC model.")
    else:
        raise ValueError("Unsupported dataset.")

    model.to(device)
    model.eval()
    return model


# ----------- Menu Execution -----------

selected_dataset = menu("Select a dataset:", DATASETS)
selected_model = menu(f"Select a model for {selected_dataset}:", MODEL_DATASET_COMPATIBILITY[selected_dataset])

net = load_model(selected_dataset, selected_model)
print(f"Model {selected_model} for {selected_dataset} loaded and ready.")
