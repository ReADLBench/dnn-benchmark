import os
import git
import shutil
import sys

# --- Available Frameworks, Datasets, and Models ---
FRAMEWORKS = ['Pytorch', 'Tensorflow', 'Both']
DATASETS = ['CIFAR10', 'CIFAR100', 'GTSRB', 'PASCAL_VOC']

MODEL_DATASET_COMPATIBILITY = {
    'CIFAR10': ['ResNet20', 'ResNet32', 'ResNet44', 'DenseNet121', 'DenseNet161',
                'GoogLeNet', 'MobileNetV2', 'Vgg11_bn', 'Vgg13_bn'],
    'CIFAR100': ['ResNet18', 'GoogLeNet', 'DenseNet121'],
    'GTSRB': ['ResNet20', 'Vgg11_bn', 'DenseNet121'],
    'PASCAL_VOC': ['deeplabv3_resnet50'],
}

# --- Model URLs ---
model_urls = {
    'ResNet20': {'CIFAR10': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/resnet20-cifar10.git',
                 'GTSRB': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/resnet20-gtsrb.git'},
    'ResNet32': {'CIFAR10': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/resnet32-cifar10.git'},
    'ResNet44': {'CIFAR10': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/resnet44-cifar10.git'},
    'ResNet18': {'CIFAR100': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/resnet18-cifar100.git'},
    'DenseNet121': {'CIFAR10': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/densenet121-cifar10.git',
                    'CIFAR100': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/densenet121-cifar100.git',
                    'GTSRB': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/densenet121-gtsrb.git'},
    'DenseNet161': {'CIFAR10': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/densenet161-cifar10.git'},
    'GoogLeNet': {'CIFAR10': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/goolenet.git',
                  'CIFAR100': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/googlenet-cifar100.git'},
    'MobileNetV2': {'CIFAR10': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/mobilenetv2-cifar10.git'},
    'Vgg11_bn': {'CIFAR10': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/vgg-11-cifar10.git',
                 'GTSRB': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/vgg-11-gtsrb.git'},
    'Vgg13_bn': {'CIFAR10': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/vgg-13-cifar10.git'},
    'deeplabv3_resnet50': {'PASCAL_VOC': 'https://gitlab.pmcs2i.ec-lyon.fr/reliability-nn-benchmark/deeplabv3-pascal-voc.git'}
}

# --- Utility Functions ---

def get_user_choice(options, prompt):
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        try:
            idx = int(input(f"Enter a number (1-{len(options)}): "))
            if 1 <= idx <= len(options):
                return options[idx - 1]
            print("Invalid option. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_model_family(model_name):
    lower = model_name.lower()
    if "resnet" in lower:
        return "resnet"
    elif "densenet" in lower:
        return "densenet"
    elif "vgg" in lower:
        return "vgg"
    elif "googlenet" in lower:
        return "googlenet"
    elif "mobilenet" in lower:
        return "mobilenet"
    elif "deeplabv3" in lower:
        return "Deeplabv3"
    else:
        return "other"

def get_task_type(model_name):
    return "image_segmentation" if "deeplabv3" in model_name.lower() else "image_classification"

def clone_repo(git_url, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    try:
        print(f"\nCloning from {git_url} into {target_dir}...")
        git.Repo.clone_from(git_url, target_dir)
        print("✅ Clone completed successfully!\n")
    except Exception as e:
        print(f"❌ Clone failed: {e}")

def remove_other_frameworks(repo_dir, selected_framework):
    if selected_framework == "Both":
        return
    other = "Tensorflow" if selected_framework.lower() == "pytorch" else "PyTorch"
    # print(f"🧹 Removing unused '{other}' folder...")
    other_path = os.path.join(repo_dir, other)
    # print(f"Checking for '{other}' folder at {other_path}...")
    if os.path.isdir(other_path):
        try:
            shutil.rmtree(other_path)
            print(f"🧹 Removed unused '{other}' folder.")
        except Exception as e:
            print(f"⚠️ Could not remove '{other}' folder: {e}")

# --- Main Script ---

def main():
    print("==== Repository Downloader ====\n")
    framework = get_user_choice(FRAMEWORKS, "Select a framework:")
    dataset = get_user_choice(DATASETS, "Select a dataset:")
    models = MODEL_DATASET_COMPATIBILITY.get(dataset, [])

    if not models:
        print(f"❌ No models available for dataset '{dataset}'. Exiting.")
        sys.exit(1)

    model = get_user_choice(models, f"Select a model for {dataset}:")
    model_url = model_urls.get(model, {}).get(dataset)

    if not model_url:
        print(f"❌ No URL available for {model} with {dataset}. Exiting.")
        sys.exit(1)

    task_type = get_task_type(model)
    model_family = get_model_family(model)

    if framework == "Both":
        target_dir = os.path.join("models", f"{model}_{dataset}")
        clone_repo(model_url, target_dir)
    else:
        fw = "pytorch" if framework == "Pytorch" else "tensorflow"
        target_dir = os.path.join(
            fw, "gpu", task_type, dataset, "fp32", model_family, model
        )
        clone_repo(model_url, target_dir)
        remove_other_frameworks(target_dir, fw)

if __name__ == "__main__":
    main()
