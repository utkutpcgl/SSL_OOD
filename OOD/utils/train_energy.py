# -*- coding: utf-8 -*-
from cProfile import label
from pickle import TRUE
from pyexpat import model
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from pathlib import Path
from typing import OrderedDict
import progressbar

# from utils.images_loader300k import RImages_300K
from utils.tinyimages_80mn_loader import TinyImages

# %run MyOtherNotebook.ipynb To import from other ipynb files.
# from models.wrn import WideResNet

# input image size settings
with open("ft_hyperparameters.yaml", "r") as reader:
    HYPS = yaml.safe_load(reader)

# Optimization hyps
batch_size_ID = HYPS["batch_size_ID"]  # Energy loss scale.
batch_size_OD = HYPS["batch_size_OD"]  # Energy loss scale.
batch_size_TEST = HYPS["batch_size_TEST"]  # Energy loss scale.
epochs = HYPS["epochs"]  # Energy loss scale.
initial_lr = HYPS["initial_lr"]  # Energy loss scale.
final_lr = HYPS["final_lr"]  # Energy loss scale.
m_out = HYPS["m_out"]  # Energy loss scale.
m_in = HYPS["m_in"]  # Energy loss scale.
lambda_energy = HYPS["lambda"]  # Energy loss scale.
decay = HYPS["decay"]  # Energy loss scale.
momentum = HYPS["momentum"]  # Energy loss scale.
energy_temperature = HYPS["energy_temperature"]  # Energy loss scale.

# Checkpoints
SAVE = "./checkpoints/"
LOAD = "./checkpoints/pretrained/"
TEST = False
NGPU = 1
WORKERS = 8
SEED = 1

SAVE_INFO = "energy_ft"

DEVICE = "cuda:0"
FREEZE_RESNET = False
NUMBER_OF_CLASSES = 10

MODEL_SETTINGS = ["BYOL", "ImageNet_Pretrained", "DINO", "BarlowTwins"]

CIFAR10_WEIGHTS = {
    "BYOL": "/home/utku/Documents/repos/SSL_OOD/OOD/checkpoints/full_cifar10_train/BYOL_trained_on_cifar10.pt",
    "BarlowTwins": "/home/utku/Documents/repos/SSL_OOD/OOD/checkpoints/full_cifar10_train/BarlowTwins_trained_on_cifar10.pt",
    "DINO": "/home/utku/Documents/repos/SSL_OOD/OOD/checkpoints/full_cifar10_train/DINO_trained_on_cifar10.pt",
    "ImageNet_Pretrained": "/home/utku/Documents/repos/SSL_OOD/OOD/checkpoints/full_cifar10_train/ImageNet_Pretrained_trained_on_cifar10.pt",
}

REGULAR_WEIGHTS = {
    "BYOL": "/home/utku/Documents/repos/SSL_OOD/SSL_checkpoints/resnet50_byol_imagenet2012.pth.tar",
    "BarlowTwins": "/home/utku/Documents/repos/SSL_OOD/SSL_checkpoints/barlowT_resnet50.pth",
    "DINO": "/home/utku/Documents/repos/SSL_OOD/SSL_checkpoints/dino_resnet50_pretrain.pth",
    "ImageNet_Pretrained": "IMAGENET1K_V2",
}

# MEAN and standard deviation of channels of CIFAR-10 images
MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
STD = [x / 255 for x in [63.0, 62.1, 66.7]]


def get_raw_network(model_setting, cifar10_pretrained_bool):
    if cifar10_pretrained_bool:
        weight_path = CIFAR10_WEIGHTS[model_setting]
    else:
        weight_path = REGULAR_WEIGHTS[model_setting]

    ResNet = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    number_of_input_features = ResNet.fc.in_features
    if model_setting == "ImageNet_Pretrained":
        ResNet.fc = torch.nn.Linear(number_of_input_features, NUMBER_OF_CLASSES)
        if cifar10_pretrained_bool:
            ResNet.load_state_dict(torch.load(weight_path)["net"])
    else:
        if cifar10_pretrained_bool:
            ResNet.fc = torch.nn.Linear(number_of_input_features, NUMBER_OF_CLASSES)
            # Load state dict after adding final linear layer.
            ResNet.load_state_dict(torch.load(weight_path)["net"])
        else:
            ResNet.fc = torch.nn.Identity()
            if model_setting == "BYOL":
                state_dict = torch.load(weight_path)["online_backbone"]
                correct_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    correct_state_dict[name] = v
                ResNet.load_state_dict(correct_state_dict)
            elif model_setting == "BarlowTwins" or model_setting == "DINO":
                ResNet.load_state_dict(torch.load(weight_path))
            else:
                raise Exception("Wrong model setting entered.")
            ResNet.fc = torch.nn.Linear(number_of_input_features, NUMBER_OF_CLASSES)
    if NGPU > 1:
        ResNet = torch.nn.DataParallel(ResNet, device_ids=list(range(NGPU)))
    return ResNet.to(DEVICE)


def freeze_resnet(ResNet):
    # Freeze ResNet
    for name, param in ResNet.named_parameters():
        if name in ["fc.weight", "fc.bias"]:
            print(f"Set final {name} layer trainable.")
            param.requires_grad = True
        else:
            param.requires_grad = False  # Freeze layers


def get_data_loaders():
    Path(SAVE).mkdir(exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    train_data_in = datasets.CIFAR10("../cifar-10", train=True, transform=train_transform)
    test_data = datasets.CIFAR10("../cifar-10", train=False, transform=test_transform)
    ood_data = TinyImages(
        transform=transforms.Compose(
            [
                # transforms.ToTensor(),
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    )

    train_loader_in = torch.utils.data.DataLoader(
        train_data_in, batch_size=batch_size_ID, shuffle=True, num_workers=WORKERS, pin_memory=True
    )

    train_loader_out = torch.utils.data.DataLoader(
        ood_data, batch_size=batch_size_OD, shuffle=False, num_workers=WORKERS, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size_TEST, shuffle=False, num_workers=WORKERS, pin_memory=True
    )
    return train_loader_in, train_loader_out, test_loader


def get_avg_energy(dataloader, network):
    total_energy = 0
    total_num_samples = 0
    len_of_iterations = len(dataloader)
    for (data, label) in tqdm(dataloader, total=len_of_iterations):
        data = data.to(DEVICE)
        output = network(data)
        Energy_in = -sum(torch.logsumexp(output, dim=1)).cpu().item()
        total_energy += Energy_in
        total_num_samples += len(data)
    avg_energy = total_energy/total_num_samples 
    return avg_energy


def get_average_energy_of_dataset(out_dataloader, in_dataloader, network):
    """Get the average energy to be used as m_in or m_out for the loss."""
    averge_energy_out = get_avg_energy(out_dataloader, network)
    averge_energy_in = get_avg_energy(in_dataloader, network)
    return averge_energy_out, averge_energy_in


def train(net, optimizer, scheduler, train_loader_in, train_loader_out, state):
    net.train()  # enter train mode
    loss_avg = 0.0
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    total_tqdms = len(train_loader_in)
    for in_set, out_set in tqdm(zip(train_loader_in, train_loader_out), total=total_tqdms):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        data, target = data.to(DEVICE), target.to(DEVICE)
        # forward
        x = net(data)
        # backward
        scheduler.step()  # update the steps of cosine annealing every step (over batches.)
        optimizer.zero_grad()
        loss = F.cross_entropy(x[: len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        Ec_out = -torch.logsumexp(x[len(in_set[0]) :], dim=1)
        Ec_in = -torch.logsumexp(x[: len(in_set[0])], dim=1)
        loss += lambda_energy * (
            torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean()
        )
        trial_loss = 10 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())
        loss.backward()
        optimizer.step()
        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    state["train_loss"] = loss_avg


# test function
def test(net, test_loader, state):
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)
            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
            # test loss average
            loss_avg += float(loss.data)

    state["test_loss"] = loss_avg / len(test_loader)
    state["test_accuracy"] = correct / len(test_loader.dataset)


def train_loop(
    net,
    optimizer,
    scheduler,
    train_loader_in,
    train_loader_out,
    test_loader,
    model_save_path_str,
    csv_file_path_str,
):
    print("Beginning Training\n")
    # Main loop
    state = {}
    for epoch in range(0, epochs):
        state["epoch"] = epoch
        begin_epoch = time.time()

        train(net, optimizer, scheduler, train_loader_in, train_loader_out, state)
        test(net, test_loader, state)

        # Save model
        torch.save(net.state_dict(), model_save_path_str)

        # Show results
        with open(csv_file_path_str, "a") as f:
            f.write(
                "%03d,%05d,%0.6f,%0.5f,%0.2f\n"
                % (
                    (epoch + 1),
                    time.time() - begin_epoch,
                    state["train_loss"],
                    state["test_loss"],
                    100 - 100.0 * state["test_accuracy"],
                )
            )

        print(
            "Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}".format(
                (epoch + 1),
                int(time.time() - begin_epoch),
                state["train_loss"],
                state["test_loss"],
                100 - 100.0 * state["test_accuracy"],
            )
        )


def create_files_and_dirs(model_setting: str, freeze_resnet: bool, cifar10_pretrained_bool: bool):
    # Make save directory
    Path(SAVE).mkdir(exist_ok=True)
    prefix_path_str = os.path.join(
        SAVE,
        model_setting
        + f"{freeze_resnet*'_frozen'+cifar10_pretrained_bool*'_cifar10_pretrained'+f'_s{SEED}_{SAVE_INFO}'}",
    )
    csv_file_path_str = prefix_path_str + "_training_results.csv"
    model_save_path_str = prefix_path_str + "_model.pt"
    with open(csv_file_path_str, "w") as f:
        f.write("epoch,time(s),train_loss,test_loss,test_error(%)\n")
    return csv_file_path_str, model_save_path_str


def train_energy(model_setting, cifar10_pretrained_bool, freeze_backbone):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = True  # fire on all cylinders
    train_loader_in, train_loader_out, test_loader = get_data_loaders()
    ResNet = get_raw_network(model_setting=model_setting, cifar10_pretrained_bool=cifar10_pretrained_bool)
    if freeze_backbone:
        freeze_resnet(ResNet=ResNet)
    optimizer = torch.optim.SGD(ResNet.parameters(), initial_lr, momentum=momentum, weight_decay=decay, nesterov=True)
    total_num_steps = epochs * len(train_loader_in)  # BUG this here might be wrong.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_num_steps, eta_min=final_lr)
    csv_file_path_str, model_save_path_str = create_files_and_dirs(
        model_setting=model_setting, freeze_resnet=freeze_backbone, cifar10_pretrained_bool=cifar10_pretrained_bool
    )
    train_loop(
        ResNet,
        optimizer,
        scheduler,
        train_loader_in,
        train_loader_out,
        test_loader,
        model_save_path_str,
        csv_file_path_str,
    )


def get_average_energy_of_network(model_setting):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = True  # fire on all cylinders
    train_loader_in, train_loader_out, test_loader = get_data_loaders()
    log_path = Path(f"checkpoints/full_cifar10_train/{model_setting}_trained_on_cifar10.txt")
    with open(str(log_path), "a") as logger:
        logger.write("\n\nStart calculating the avg. energy for the model: " + model_setting)
    print("Start calculating the avg. energy for the model: " + model_setting)
    ResNet = get_raw_network(model_setting=model_setting, cifar10_pretrained_bool=True)
    in_avg_energy, out_avg_energy = get_average_energy_of_dataset(
        out_dataloader=train_loader_out, in_dataloader=train_loader_in, network=ResNet
    )
    energy_report = f"\ncifar10 average energy is {in_avg_energy}, tiny_images average energy is {out_avg_energy}"
    print(energy_report)
    with open(str(log_path), "a") as logger:
        logger.write(energy_report + "\n")


def main():
    train_energy(model_setting="BYOL", cifar10_pretrained_bool=True, freeze_backbone=False)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.8, device=DEVICE)
    for model_setting in MODEL_SETTINGS:
        get_average_energy_of_network(model_setting=model_setting)
