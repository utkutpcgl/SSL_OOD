import numpy as np
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from PIL import Image as PILImage
import torchvision
import yaml
from pathlib import Path
from utils.train_energy import (
    batch_size_TEST,
    energy_temperature,
    WORKERS,
    NGPU,
    NUMBER_OF_CLASSES,
    get_raw_network,
    MEAN,
    STD,
)


# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.score_calculation as lib


CIFAR10_WEIGHTS = Path("/home/utku/Documents/repos/SSL_OOD/OOD/checkpoints/full_cifar10_train")
FULL_ENERGY_FT_WEIGHTS = Path("/home/utku/Documents/repos/SSL_OOD/OOD/checkpoints/full_energy_finetune")
LINEAR_ENERGY_FT_WEIGHTS = Path("/home/utku/Documents/repos/SSL_OOD/OOD/checkpoints/linear_energy_finetune")
SSL_WEIGHTS = Path("/home/utku/Documents/repos/SSL_OOD/SSL_checkpoints")

TEST_TRANSFORM = trn.Compose([trn.ToTensor(), trn.Normalize(MEAN, STD)])


def get_weights_list_in_path(source_path: Path):
    path_list = []
    for path in source_path.iterdir():
        if path.suffix == ".pt":
            path_list.append(path)
    return path_list


def get_energy_fine_tuned_network(weight_path, device):
    """no_ft_weight_bool: if the weights are the SSL or
    ImageNet weights without fine tuning, special treatment is neccessary.
    """
    ResNet = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    number_of_input_features = ResNet.fc.in_features
    ResNet.fc = torch.nn.Linear(number_of_input_features, NUMBER_OF_CLASSES)
    ResNet.load_state_dict(torch.load(weight_path))
    if NGPU > 1:
        ResNet = torch.nn.DataParallel(ResNet, device_ids=list(range(NGPU)))
    if NGPU > 0:
        ResNet.cuda()
    cudnn.benchmark = True  # fire on all cylinders
    return ResNet.to(device)


def get_test_loader() -> int:
    """Return ood_num_examples which is used to determine number of OOD samples."""
    test_data = dset.CIFAR10("/home/utku/Documents/repos/SSL_OOD/cifar-10", train=False, transform=TEST_TRANSFORM)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size_TEST, shuffle=False, num_workers=WORKERS, pin_memory=True
    )
    ood_num_examples = len(test_data) // 5
    return test_loader, ood_num_examples


def get_ood_scores(loader, ResNet, ood_num_examples, in_dist=False):
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    """Get scores would be a better name as the method is also used for correct classifications."""
    _score = []  # The ID data energy scores are stored here.
    # The scores below are used to find the error rate at the original in distribution task.
    # Number of right/ predictions are used rather than their sofmtax score.
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // batch_size_TEST and in_dist is False:
                break

            data = data.cuda()

            output = ResNet(data)
            smax = to_np(F.softmax(output, dim=1))

            _score.append(-to_np((energy_temperature * torch.logsumexp(output / energy_temperature, dim=1))))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)
                # Calculate the scores, only the lenght of those score lists are used.
                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list):
    aurocs, auprs, fprs = [], [], []
    out_score = get_ood_scores(ood_loader)
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0])
    auprs.append(measures[1])
    fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    print_measures(auroc, aupr, fpr)


def get_texture_results(in_score):
    auroc_list, aupr_list, fpr_list = [], [], []
    ood_data = dset.ImageFolder(
        root="/home/utku/Documents/repos/SSL_OOD/dtd-r1.0.1/Describable_Textures_Dataset/images",
        transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(MEAN, STD)]),
    )
    ood_loader = torch.utils.data.DataLoader(
        ood_data, batch_size=batch_size_TEST, shuffle=True, num_workers=4, pin_memory=True
    )
    print("\n\nTexture Detection")
    get_and_print_results(ood_loader)
    print("\n\nMean Test Results!!!!!")
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list))


def get_other_results(ood_num_examples):
    auroc_list, aupr_list, fpr_list = [], [], []
    # /////////////// Uniform Noise ///////////////
    dummy_targets = torch.ones(ood_num_examples)
    ood_data = torch.from_numpy(
        np.random.uniform(size=(ood_num_examples, 3, 32, 32), low=-1.0, high=1.0).astype(np.float32)
    )
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size_TEST, shuffle=True)

    print("\n\nUniform[-1,1] Noise Detection")
    get_and_print_results(ood_loader)

    # /////////////// Arithmetic Mean of Images ///////////////
    ood_data = dset.CIFAR100("/home/utku/Documents/repos/SSL_OOD/cifar-100", train=False, transform=TEST_TRANSFORM)

    class AvgOfPair(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.shuffle_indices = np.arange(len(dataset))
            np.random.shuffle(self.shuffle_indices)

        def __getitem__(self, i):
            random_idx = np.random.choice(len(self.dataset))
            while random_idx == i:
                random_idx = np.random.choice(len(self.dataset))

            return self.dataset[i][0] / 2.0 + self.dataset[random_idx][0] / 2.0, 0

        def __len__(self):
            return len(self.dataset)

    ood_loader = torch.utils.data.DataLoader(
        AvgOfPair(ood_data), batch_size=batch_size_TEST, shuffle=True, num_workers=WORKERS, pin_memory=True
    )

    print("\n\nArithmetic Mean of Random Image Pair Detection")
    get_and_print_results(ood_loader)

    # /////////////// Geometric Mean of Images ///////////////
    ood_data = dset.CIFAR100(
        "/home/utku/Documents/repos/SSL_OOD/cifar-100", train=False, download=True, transform=trn.ToTensor()
    )

    class GeomMeanOfPair(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.shuffle_indices = np.arange(len(dataset))
            np.random.shuffle(self.shuffle_indices)

        def __getitem__(self, i):
            random_idx = np.random.choice(len(self.dataset))
            while random_idx == i:
                random_idx = np.random.choice(len(self.dataset))

            return trn.Normalize(MEAN, STD)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0

        def __len__(self):
            return len(self.dataset)

    ood_loader = torch.utils.data.DataLoader(
        GeomMeanOfPair(ood_data), batch_size=batch_size_TEST, shuffle=True, num_workers=WORKERS, pin_memory=True
    )

    print("\n\nGeometric Mean of Random Image Pair Detection")
    get_and_print_results(ood_loader)

    # /////////////// Jigsaw Images ///////////////

    ood_loader = torch.utils.data.DataLoader(
        ood_data, batch_size=batch_size_TEST, shuffle=True, num_workers=WORKERS, pin_memory=True
    )

    jigsaw = lambda x: torch.cat(
        (
            torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1), x[:, 16:, :16]), 2),
            torch.cat((x[:, 16:, 16:], torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
        ),
        1,
    )

    ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), jigsaw, trn.Normalize(MEAN, STD)])

    print("\n\nJigsawed Images Detection")
    get_and_print_results(ood_loader)

    # /////////////// Speckled Images ///////////////

    speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
    ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), speckle, trn.Normalize(MEAN, STD)])

    print("\n\nSpeckle Noised Images Detection")
    get_and_print_results(ood_loader)

    # /////////////// Pixelated Images ///////////////

    pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.BOX).resize((32, 32), PILImage.BOX)
    ood_loader.dataset.transform = trn.Compose([pixelate, trn.ToTensor(), trn.Normalize(MEAN, STD)])

    print("\n\nPixelate Detection")
    get_and_print_results(ood_loader)

    # /////////////// RGB Ghosted/Shifted Images ///////////////

    rgb_shift = lambda x: torch.cat(
        (x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])), x[2:, :, :], x[0:1, :, :]), 0
    )
    ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), rgb_shift, trn.Normalize(MEAN, STD)])

    print("\n\nRGB Ghosted/Shifted Image Detection")
    get_and_print_results(ood_loader)

    # /////////////// Inverted Images ///////////////

    # not done on all channels to make image ood with higher probability
    invert = lambda x: torch.cat(
        (
            x[0:1, :, :],
            1
            - x[
                1:2,
                :,
            ],
            1 - x[2:, :, :],
        ),
        0,
    )
    ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), invert, trn.Normalize(MEAN, STD)])

    print("\n\nInverted Image Detection")
    get_and_print_results(ood_loader)

    # /////////////// Mean Results ///////////////

    print("\n\nMean Validation Results")
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list))


def test_loop(energy_ft_bool: False):

    if energy_ft_bool:
        get_energy_fine_tuned_network()
    else:
        get_raw_network()

    test_loader, ood_num_examples = get_test_loader()
    in_score, right_score, wrong_score = get_ood_scores(
        test_loader, ood_num_examples, in_dist=True
    )  # Only in_score is important here.
    get_texture_results(in_score)
    get_other_results(ood_num_examples)
