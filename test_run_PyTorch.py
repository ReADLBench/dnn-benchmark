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

import os

import shutil
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB, CIFAR10, CIFAR100, VOCSegmentation
from torchvision.transforms.v2 import ToTensor,Resize,Compose,ColorJitter,RandomRotation,AugMix,GaussianBlur,RandomEqualize,RandomHorizontalFlip,RandomVerticalFlip

import csv
from tqdm import tqdm

import random

DATASETS = ['CIFAR10', 'CIFAR100', 'GTSRB', 'PASCAL_VOC']

DATASET_PATH = f'Datasets/'
BATCH_SIZE = 64

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


def load_model(dataset, model_name, device):
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
            weight_path = './pytorch/gpu/image_segmentation/PASCAL_VOC/fp32/DeepLabV3/deeplabv3_resnet50/PyTorch/deeplabv3_resnet50.pth'
            model = torch.load(weight_path, map_location=device, weights_only =False)
        elif model_name == 'DeepLabV3_resnet50_state_dict':
            model = deeplabv3_resnet50(weights=None, num_classes=21, aux_loss=True)
            weight_path = './pytorch/gpu/image_segmentation/PASCAL_VOC/fp32/DeepLabV3/deeplabv3_resnet50/PyTorch/deeplabv3_resnet50.pt'
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

def get_loader(network_name: str,
               batch_size: int,
               image_per_class: int = None,
               dataset_name: str = None,
               network: torch.nn.Module = None) -> DataLoader:
    """
    Return the loader corresponding to a given network and with a specific batch size
    :param network_name: The name of the network
    :param batch_size: The batch size
    :param image_per_class: How many images to load for each class
    :param network: Default None. The network used to select the image per class. If not None, select the image_per_class
    that maximize this network accuracy. If not specified, images are selected at random
    :return: The DataLoader
    """
    if 'CIFAR10' == dataset_name:
        print('Loading CIFAR10 dataset')
        train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
                                             test_image_per_class=image_per_class)
        
    elif 'CIFAR100' == dataset_name:
        print('Loading CIFAR100 dataset')
        train_loader, _, loader = Load_CIFAR100_datasets(test_batch_size=batch_size,
                                             test_image_per_class=image_per_class)
        
    elif 'GTSRB' == dataset_name:
        print('Loading GTSRB dataset')
        train_loader, _, loader = Load_GTSRB_datasets(test_batch_size=batch_size,
                                             test_image_per_class=image_per_class)
        
    elif 'PASCAL_VOC' == dataset_name:
        print('Loading PASCAL_VOC dataset')
        loader = Load_PASCAL_VOC_datasets(test_batch_size=batch_size,
                                             test_image_per_class=image_per_class)
        
    else:
        print('no dataset specified')
        exit()


    print(f'Batch size:\t\t{batch_size} \nNumber of batches:\t{len(loader)}')

    return loader


def Load_PASCAL_VOC_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):
    

    val_transform = transforms.Compose([
                transforms.Resize((520, 520)),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
    
    t_transform = transforms.Compose([
        transforms.v2.ToImage(),
        Resize((520, 520), interpolation =  transforms.v2.InterpolationMode.NEAREST),
        
    ])
   
    
    val_dataset = VOCSegmentation(root=DATASET_PATH,
                                year='2012',
                                image_set='val',
                                download=True,
                                transform=val_transform,
                                target_transform = t_transform)

    

    
    test_loader = DataLoader(dataset=val_dataset,
                                batch_size=test_batch_size,
                                shuffle=False
                            )
    
   

    print('PASCA VOC Dataset loaded')
    
    
    return test_loader

def Load_GTSRB_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):
    
    train_transforms = Compose([
    ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
    RandomEqualize(0.4),
    AugMix(),
    RandomHorizontalFlip(0.3),
    RandomVerticalFlip(0.3),
    GaussianBlur((3,3)),
    RandomRotation(30),
    
    Resize([50,50]),
    ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))
    
    ])

    validation_transforms = Compose([
        Resize([50, 50]),
        ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    ])

    train_dataset = GTSRB(root=DATASET_PATH,
                            split='train',
                            download=True,
                            transform=train_transforms)
    test_dataset = GTSRB(root=DATASET_PATH,
                            split='test',
                            download=True,
                            transform=validation_transforms)



    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * 0.8)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                                lengths=[train_split_length, val_split_length],
                                                                generator=torch.Generator().manual_seed(1234))
    # DataLoader is used to load the dataset
    # for training
    train_loader = DataLoader(dataset=train_subset,
                                                batch_size=train_batch_size,
                                                shuffle=True)
    val_loader = DataLoader(dataset=val_subset,
                                                batch_size=train_batch_size,
                                                shuffle=True)  

    test_loader = DataLoader(dataset=test_dataset,
                                                batch_size=test_batch_size,
                                                shuffle=False)

    print('GTSRB Dataset loaded')
        
    return train_loader, val_loader, test_loader

def Load_CIFAR100_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    train_dataset = CIFAR100(DATASET_PATH, train=True, transform=transform, download=True)
    test_dataset = CIFAR100(DATASET_PATH, train=False, transform=transform, download=True)

    train_split = 0.8
    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, lengths=[train_split_length, val_split_length], generator=torch.Generator().manual_seed(1234))

    train_loader = DataLoader(dataset=train_subset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
    
    print('CIFAR100 Dataset loaded')

    return train_loader, val_loader, test_loader

def load_CIFAR10_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                       # Crop the image to 32x32
        transforms.RandomHorizontalFlip(),                                          # Data Augmentation
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),                                                  # Crop the image to 32x32
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])

    train_dataset = CIFAR10(DATASET_PATH,
                            train=True,
                            transform=transform_train,
                            download=True)
    test_dataset = CIFAR10(DATASET_PATH,
                           train=False,
                           transform=transform_test,
                           download=True)

    if test_image_per_class is not None:
        selected_test_list = []
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                selected_test_list.append(test_image)
                image_class_counter[test_image[1]] += 1
        test_dataset = selected_test_list

    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                             lengths=[train_split_length, val_split_length],
                                                             generator=torch.Generator().manual_seed(1234))
    # DataLoader is used to load the dataset
    # for training
    train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                               batch_size=train_batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                             batch_size=train_batch_size,
                                             shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    print('CIFAR10 Dataset loaded')

    return train_loader, val_loader, test_loader

def clean_inference(network, loader, device, network_name):
       
    clean_output_scores = list()
    clean_output_indices = list()
    clean_labels = list()

    counter = 0
    with torch.no_grad():
        pbar = tqdm(loader,
                colour='green',
                desc=f'Clean Run',
                ncols=shutil.get_terminal_size().columns)

        dataset_size = 0
        
        for batch_id, batch in enumerate(pbar):
            data, label = batch
            dataset_size = dataset_size + len(label)
            data = data.to(device)
            
            network_output = network(data)
            prediction = torch.topk(network_output, k=1)
            scores = network_output.cpu()
            indices = [int(fault) for fault in prediction.indices]
            
            clean_output_scores.append(scores)
            clean_output_indices.append(indices)
            clean_labels.append(label)
            
            counter = counter + 1


        elementwise_comparison = [label != index for labels, indices in zip(clean_labels, clean_output_indices) for label, index in zip(labels, indices)]          
        # Count the number of different elements
        num_different_elements = elementwise_comparison.count(True)
        
        print(f'device: {device}')
        print(f'network: {network_name}')
        print(f"The DNN wrong predicions are: {num_different_elements}")
        accuracy= (1 - num_different_elements/dataset_size)*100
        print(f"The final accuracy is: {accuracy}%")




def image_segmentation_clean_inference(network, loader, device, network_name):
    

    correctPixels = 0
    numclass = 21
    same_pixels = []

    pbar = tqdm(loader, bar_format="{l_bar}{bar:10}{r_bar}")  # progress bar
    batch_id = 0
    totalIOUs = []
    batch_files = []  # Lista per tenere traccia dei file salvati
    batch_sizes = []  # Lista per salvare le dimensioni delle batch

    
    # ------- to check if golden_output matrix is correct ------------
    # golden_output = np.load('results_segmentation/all_batches.npy')
    # golden_output = torch.tensor(golden_output)
    # ----------------------------------------------------------------
    
    with torch.no_grad(): 
        for img, label in pbar:
            img = img.to(device)
            label = label.to(device)

            batch_size = img.size(0)
            
            output = network(img)["out"]
            pred = output.argmax(axis=1)

            label = label.squeeze(1)

            # pixel accuracy
            diff = pred == label
            correctPixels = diff.sum(axis=[1,2])
            same_pixels.append(correctPixels)
            
            ious = torch.zeros((numclass, batch_size))
                # print(ious.shape)
            for cls in range(numclass):
                clsPred = pred == cls
                clsLab = label == cls
                inter = torch.logical_and(clsPred, clsLab).sum(axis=[1,2])
                union = torch.logical_or(clsPred, clsLab).sum(axis=[1,2])
                iou = inter/union
                # print(iou.shape)
                ious[cls] = iou
            totalIOUs.append(ious)
            
            batch_id += 1
        

    # total pixel accuracy
    total_same_pixels = torch.concat(same_pixels)
    print(total_same_pixels.shape)
    print('Pixel Accuracy',total_same_pixels.sum() / (total_same_pixels.shape[0] * 520*520))
    
    
    # table with IOUS. axis 0 has the categories, axis 1 has the images
    iousT = torch.concat([*totalIOUs], axis=1)

    # mIOU per image
    mIOU_image = torch.nanmean(iousT, axis=0)
    print(mIOU_image)

    # mIOU for the network
    mIOU = torch.mean(mIOU_image)
    print('Mean Intersection Over Union',mIOU)
        

# ----------- Menu Execution -----------




def main():
    
   

    selected_dataset = menu("Select a dataset:", DATASETS)
    selected_model = menu(f"Select a model for {selected_dataset}:", MODEL_DATASET_COMPATIBILITY[selected_dataset])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    net = load_model(selected_dataset, selected_model, device)
    
    print(f"Model {selected_model} for {selected_dataset} loaded and ready.")
    
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        print(f"Created dataset directory at {DATASET_PATH}")
    
    loader = get_loader(network_name=net,
                        batch_size=BATCH_SIZE,
                        dataset_name=selected_dataset)
    
    print('clean inference accuracy test:')
    if selected_dataset == 'PASCAL_VOC':
        image_segmentation_clean_inference(net, loader, device, selected_model)
    else:
        clean_inference(net, loader, device, selected_model)

if __name__ == '__main__':
    main()