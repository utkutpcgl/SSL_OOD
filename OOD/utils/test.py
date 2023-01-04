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
    MODEL_SETTINGS,
    batch_size_TEST,
    energy_temperature,
    WORKERS,
    NGPU,
    NUMBER_OF_CLASSES,
    get_raw_network,
    MEAN,
    STD,
    SEED,
)


# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import (
        show_performance,
        get_measures,
        print_measures,
        print_and_log,
    )  # change prints to logs
    import utils.score_calculation as lib

# Log paths
TRAINED_LOG_PATH = Path("/home/utku/Documents/repos/SSL_OOD/OOD/test/trained/")
BASE_LOG_PATH = Path("/home/utku/Documents/repos/SSL_OOD/OOD/test/base/")
DEVICE = "cuda:0"

# Trained weight paths
CIFAR10_WEIGHTS = Path("/home/utku/Documents/repos/SSL_OOD/OOD/checkpoints/full_cifar10_train")
FULL_ENERGY_FT_WEIGHTS = Path("/home/utku/Documents/repos/SSL_OOD/OOD/checkpoints/full_energy_finetune")
LINEAR_ENERGY_FT_WEIGHTS = Path("/home/utku/Documents/repos/SSL_OOD/OOD/checkpoints/linear_energy_finetune")
TRAINED_WEIGHT_PATHS_LIST = [CIFAR10_WEIGHTS, FULL_ENERGY_FT_WEIGHTS, LINEAR_ENERGY_FT_WEIGHTS]
SSL_WEIGHTS = Path("/home/utku/Documents/repos/SSL_OOD/SSL_checkpoints")

# Data transforms.
TEST_TRANSFORM = trn.Compose([trn.ToTensor(), trn.Normalize(MEAN, STD)])


def mkdirs():
    TRAINED_LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
    BASE_LOG_PATH.parent.mkdir(exist_ok=True, parents=True)


# GETTING THE NETWORK


def get_weights_list_in_path(source_path: Path):
    path_list = []
    for path in source_path.iterdir():
        if path.suffix == ".pt":
            path_list.append(path)
    return path_list


def get_all_trained_weights():
    trained_weights = []
    for trained_weight_path in TRAINED_WEIGHT_PATHS_LIST:
        trained_weights.append(get_weights_list_in_path(trained_weight_path))
    return trained_weights


def get_trained_network(weight_path):
    """Get the network based on the specifications.
    Args:
        weight_path (str): Path of the weight of the target model.
    """
    ResNet = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    number_of_input_features = ResNet.fc.in_features
    ResNet.fc = torch.nn.Linear(number_of_input_features, NUMBER_OF_CLASSES)
    ResNet.load_state_dict(torch.load(weight_path))
    if NGPU > 1:
        ResNet = torch.nn.DataParallel(ResNet, device_ids=list(range(NGPU)))
    return ResNet.to(DEVICE)


def get_base_network(model_setting):
    """Get the base model to test energy OOD without any training on CIFAR10 or energy ft."""
    ResNet = get_raw_network(model_setting, cifar10_pretrained_bool=False)
    return ResNet


def get_test_loader() -> int:
    """Return ood_num_examples which is used to determine number of OOD samples."""
    test_data = dset.CIFAR10("/home/utku/Documents/repos/SSL_OOD/cifar-10", train=False, transform=TEST_TRANSFORM)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size_TEST, shuffle=False, num_workers=WORKERS, pin_memory=True
    )
    ood_num_examples = len(test_data) // 5
    return test_loader, ood_num_examples


# CALCULATE AND DISPLAY SCORES


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

            data = data.to(DEVICE)

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


def get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, ood_num_examples, log_path):
    aurocs, auprs, fprs = [], [], []
    out_score = get_ood_scores(ood_loader, ResNet, ood_num_examples)
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0])
    auprs.append(measures[1])
    fprs.append(measures[2])
    print_and_log(in_score[:3], out_score[:3], log_path)
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    print_measures(auroc, aupr, fpr, log_path=log_path)


def show_right_wrong_perf(right_score, wrong_score, log_path):
    num_right = len(right_score)
    num_wrong = len(wrong_score)
    print_and_log("Error Rate {:.2f}".format(100 * num_wrong / (num_wrong + num_right)), log_path)
    # /////////////// Error Detection ///////////////
    print_and_log("\n\nError Detection", log_path)
    show_performance(wrong_score, right_score, log_path)


def get_texture_results(ood_num_examples, in_score, ResNet, log_path):
    auroc_list, aupr_list, fpr_list = [], [], []
    ood_data = dset.ImageFolder(
        root="/home/utku/Documents/repos/SSL_OOD/dtd-r1.0.1/Describable_Textures_Dataset/images",
        transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(MEAN, STD)]),
    )
    ood_loader = torch.utils.data.DataLoader(
        ood_data, batch_size=batch_size_TEST, shuffle=True, num_workers=4, pin_memory=True
    )
    print_and_log("\n\nTexture Detection", log_path)
    get_and_print_results(
        ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, ood_num_examples, log_path=log_path
    )
    print_and_log("\n\nMean Test Results!!!!!", log_path)
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), log_path=log_path)


def get_other_results(ood_num_examples, in_score, ResNet, log_path):
    auroc_list, aupr_list, fpr_list = [], [], []
    # /////////////// Uniform Noise ///////////////
    dummy_targets = torch.ones(ood_num_examples)
    ood_data = torch.from_numpy(
        np.random.uniform(size=(ood_num_examples, 3, 32, 32), low=-1.0, high=1.0).astype(np.float32)
    )
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size_TEST, shuffle=True)
    print_and_log("\n\nUniform[-1,1] Noise Detection", log_path)
    get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, ood_num_examples, log_path)

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
    print_and_log("\n\nArithmetic Mean of Random Image Pair Detection", log_path)
    get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, ood_num_examples, log_path)

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
    print_and_log("\n\nGeometric Mean of Random Image Pair Detection", log_path)
    get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, ood_num_examples, log_path)

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
    print_and_log("\n\nJigsawed Images Detection", log_path)
    get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, ood_num_examples, log_path)

    # /////////////// Speckled Images ///////////////
    speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
    ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), speckle, trn.Normalize(MEAN, STD)])
    print_and_log("\n\nSpeckle Noised Images Detection", log_path)
    get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, ood_num_examples, log_path)

    # /////////////// Pixelated Images ///////////////
    pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.BOX).resize((32, 32), PILImage.BOX)
    ood_loader.dataset.transform = trn.Compose([pixelate, trn.ToTensor(), trn.Normalize(MEAN, STD)])
    print_and_log("\n\nPixelate Detection", log_path)
    get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, ood_num_examples, log_path)

    # /////////////// RGB Ghosted/Shifted Images ///////////////
    rgb_shift = lambda x: torch.cat(
        (x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])), x[2:, :, :], x[0:1, :, :]), 0
    )
    ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), rgb_shift, trn.Normalize(MEAN, STD)])
    print_and_log("\n\nRGB Ghosted/Shifted Image Detection", log_path)
    get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, ood_num_examples, log_path)

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
    print_and_log("\n\nInverted Image Detection", log_path)
    get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, ood_num_examples, log_path)

    # /////////////// Mean Results ///////////////
    print_and_log("\n\nMean Validation Results", log_path)
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), log_path=log_path)


# MAIN CODES TO TEST


def test_network(ResNet, log_path):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = True  # fire on all cylinders
    test_loader, ood_num_examples = get_test_loader()
    # Only in_score is important below.
    in_score, right_score, wrong_score = get_ood_scores(test_loader, ResNet, ood_num_examples, in_dist=True)
    show_right_wrong_perf(right_score, wrong_score, log_path)
    get_texture_results(ood_num_examples, in_score, ResNet, log_path=log_path)
    get_other_results(ood_num_examples, in_score, ResNet, log_path=log_path)


def test_all_networks():
    # For trained weights
    all_trained_weights = get_all_trained_weights()
    for trained_weight_path in all_trained_weights:
        log_path = f"/home/utku/Documents/repos/SSL_OOD/OOD/test/trained/{trained_weight_path.stem}_log.txt"
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)
        ResNet_trained = get_trained_network(trained_weight_path)
        test_network(ResNet_trained, log_path)

    # For base weights
    for model_setting in MODEL_SETTINGS:
        log_path = f"/home/utku/Documents/repos/SSL_OOD/OOD/test/base/{model_setting}_log.txt"
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)
        ResNet_base = get_base_network(model_setting=model_setting)
        test_network(ResNet_base, log_path)


# TODO check below
# TODO make sure the saved logs for each weight inludes the model path name.
# TODO Make sure that the plotted results are consistent with the paper (like fpr95)
# TODO make sure models are loaded correctly.
