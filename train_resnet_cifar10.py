"""Train a ResNet on CIFAR10"""
import torch
from data import PoisonedDataset, load_init_data, create_backdoor_data_loader
import os
import numpy as np

from models.vnncomp_resnet import resnet2b, resnet4b
# The following models can also be trained with, but need to modify `backdoor_model_trainer()`
# from models.badnet import BadNet
# from models.mobilenet import MobileNetV2
# from models.resnet18 import ResNet18
# from models.resnext import ResNeXt_cifar
# from models.resnet import model_resnet
# from models.densenet import Densenet_cifar_32

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_cpu = torch.device("cpu")
trigger_label = 0
poisoned_portion = 0.1
batchsize = 128
epoch = 200
lr = 0.001

"""Trigger Mark Path"""
mark_dir = None # a white square at the right bottom corner
# mark_dir = './marks/apple_white.png'
# mark_dir = './marks/apple_black.png'
# mark_dir = './marks/watermark_white.png'
# mark_dir = './marks/watermark_black.png'

alpha = 0.2 # trigger transparency

"""Saving Model Path (where to save)"""
save_path = './checkpoints/resnet4b-cifar10.pth'

"""Running Mode Selections"""
test_only = False # test the model at `pretrained_path` only when set to True; otherwise train a model
pretrained = False # resume training a model when set to True; otherwise train a new model

"""Pretrained Model Path"""
pretrained_path = './checkpoints/resnet4b-cifar10.pth'

train_data, test_data = load_init_data(dataname='cifar10', device=device_cpu, download=False, dataset_path='./dataset/')
train_data_loader, test_data_ori_loader, test_data_tri_loader = create_backdoor_data_loader('cifar10', train_data, test_data, trigger_label, poisoned_portion, batchsize, device_cpu, mark_dir=mark_dir, alpha=alpha)

import os
import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from models import BadNet
from utils import print_model_perform


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss().cuda()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        print("automatically assign mse loss function to you...")
        criterion = nn.MSELoss().cuda()

    return criterion


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer


def backdoor_model_trainer(dataname, train_data_loader, test_data_ori_loader, test_data_tri_loader, trigger_label, epoch, batch_size, loss_mode, optimization, lr, print_perform_every_epoch, basic_model_path, device):
    # badnet = BadNet(3, 10).to(device)
    # badnet = resnet2b().to(device)
    badnet = resnet4b().to(device)
    # badnet = ResNet18().to(device)
    # badnet = model_resnet().to(device)

    if pretrained:
        badnet.load_state_dict(torch.load(pretrained_path))
    else:
        for m in badnet.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    criterion = loss_picker(loss_mode)
    optimizer = optimizer_picker(optimization, badnet.parameters(), lr=lr)

    train_process = []
    print("### target label is %d, EPOCH is %d, Learning Rate is %f" % (trigger_label, epoch, lr))
    print("### Train set size is %d, ori test set size is %d, tri test set size is %d\n" % (len(train_data_loader.dataset), len(test_data_ori_loader.dataset), len(test_data_tri_loader.dataset)))
    for epo in range(epoch):
        loss = train(badnet, train_data_loader, criterion, optimizer, loss_mode)
        acc_train = eval(badnet, train_data_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)
        acc_test_ori = eval(badnet, test_data_ori_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)
        acc_test_tri = eval(badnet, test_data_tri_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)

        print("# EPOCH%d   loss: %.4f  training acc: %.4f, ori testing acc: %.4f, trigger testing acc: %.4f\n"\
              % (epo, loss.item(), acc_train, acc_test_ori, acc_test_tri))
        
        # save model 
        torch.save(badnet.state_dict(), basic_model_path)

        # save training progress
        train_process.append(( dataname, batch_size, trigger_label, lr, epo, loss.item(), acc_train, acc_test_ori, acc_test_tri))
        df = pd.DataFrame(train_process, columns=("dataname", "batch_size", "trigger_label", "learning_rate", "epoch", "loss", "train_acc", "test_ori_acc", "test_tri_acc"))
        df.to_csv("./logs/%s_train_process_trigger%d.csv" % (dataname, trigger_label), index=False, encoding='utf-8')

    return badnet


def train(model, data_loader, criterion, optimizer, loss_mode):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        output = model(batch_x.cuda()) # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y.cuda()) # mse loss
        elif loss_mode == "cross":
            loss = criterion(output, torch.argmax(batch_y.cuda(), dim=1)) # cross entropy loss

        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False):
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y_predict = model(batch_x.cuda())
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        
        batch_y = torch.argmax(batch_y.cuda(), dim=1)
        y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    if print_perform and mode != 'backdoor':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return accuracy_score(y_true.cpu(), y_predict.cpu())

def main():
    # Train
    if not test_only:
        model = backdoor_model_trainer(
                dataname='cifar10',
                train_data_loader=train_data_loader, 
                test_data_ori_loader=test_data_ori_loader,
                test_data_tri_loader=test_data_tri_loader,
                trigger_label=trigger_label,
                epoch=epoch,
                batch_size=batchsize,
                loss_mode='cross',
                optimization='sgd',
                lr=lr,
                print_perform_every_epoch=False,
                basic_model_path= save_path,
                device=device)
    else:
        model = resnet4b()
        model.load_state_dict(torch.load(pretrained_path))
    model = model.cpu()
    print("\n# evaluation")
    print("## original test data performance:")
    print_model_perform(model, test_data_ori_loader)
    print("## triggered test data performance:")
    print_model_perform(model, test_data_tri_loader)

if __name__ == "__main__":
    main()
