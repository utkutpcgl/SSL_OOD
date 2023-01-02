"""Finetune the whole network on CIFAR10."""

import torchvision
from torchvision import transforms
import torch
from typing import OrderedDict
from tqdm import tqdm
from pathlib import Path

Device = "cuda:0"

# Regular CIFAR-10 train setting for ResNet, could have been tuned for SSL.
MaxLr = 0.1  # Looked from kuangliu's code.
MinLr = 0.00001  # Default is 0 in torch cosine annealing.
MaxPeriod = 50  # Max period 200 gave best results in cosine annealing paper.
TotalEpochs = MaxPeriod

# Data transformations
# No need to resize -> CIFAR-10 is 32x32
# Was trained on wrong std previously: (0.2023, 0.1994, 0.2010)
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), [0.2471, 0.2436, 0.2616]),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), [0.2471, 0.2436, 0.2616]),
    ]
)


trainset = torchvision.datasets.CIFAR10(
    root="/home/utku/Documents/repos/SSL_OOD/cifar-10", train=False, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root="/home/utku/Documents/repos/SSL_OOD/cifar-10", train=False, download=False, transform=transform_test
)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


# Training
def train(net, epoch, optimizer, criterion, device=Device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    for inputs, targets in tqdm(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        batch_count += 1
    print(
        f"\nEpoch: {epoch}, ",
        "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (train_loss / (batch_count), 100.0 * correct / total, correct, total),
    )


def test(net, epoch, criterion, best_acc, device, model_setting):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_count += 1
    epoch_report = f"\nEpoch: {epoch}, Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
        test_loss / (batch_count),
        100.0 * correct / total,
        correct,
        total,
    )
    print(epoch_report)

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        Path("checkpoint").mkdir(exist_ok=True)
        torch.save(state, f"checkpoint/{model_setting}_trained_on_cifar10.pt")
        log_path = Path(f"checkpoint/{model_setting}_trained_on_cifar10.txt")
        with open(str(log_path), "a") as logger:
            logger.write(epoch_report + "\n")
        best_acc = acc
    return best_acc


def full_train_cifar10(net, model_setting, device=Device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=MaxLr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=MaxPeriod, eta_min=MinLr)

    best_acc = 0
    # Train loop
    for epoch in range(TotalEpochs):
        train(net, epoch, optimizer=optimizer, criterion=criterion, device=device)
        best_acc = test(net, epoch, criterion=criterion, best_acc=best_acc, device=device, model_setting=model_setting)
        scheduler.step()  # This increases current iteration (step) once every epoch (not iteration step as explained in the paper.)


if __name__ == "__main__":
    pass
