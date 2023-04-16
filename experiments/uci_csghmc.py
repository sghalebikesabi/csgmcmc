'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import scipy as sp

import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

import pdb
import os
import argparse

import wandb 

import pandas as pd

from models import *
from torch.autograd import Variable
import numpy as np
import random

parser = argparse.ArgumentParser(description='cSG-MCMC CIFAR10 Training')
parser.add_argument('--dir', type=str, default="ckpt", help='path to save checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default="data", metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--data_name', type=str, default="boston")
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--regress', type=int, default=1,
                    help='if 0, classification')
parser.add_argument('--batch-size', type=int, default=16384, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--alpha', type=float, default=0.9,
                    help='1: SGLD; <1: SGHMC')
parser.add_argument('--lr_0', type=float, default=0.004,
                    help='lr_0')
parser.add_argument('--weight_decay', type=float, default=0.4,
                    help='lr_0')
parser.add_argument('--cycle_length', type=int, default=10,
                    help='lr_0')
parser.add_argument('--offset', type=int, default=10,
                    help='lr_0')
parser.add_argument('--num_hidden', type=int, default=16,
                    help='lr_0')
parser.add_argument('--device_id',type = int, help = 'device id to use')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=100.,
                    help='temperature times dataset_size (default: 1)')

args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

wandb.init(
    project="sghmc",
    entity="sahra",
    reinit=True,
    settings=wandb.Settings(start_method="thread"),
    config=args,
)

# Data
print('==> Preparing data..')
if args.regress==0 and args.data_name != "statlog":
    if args.data_name == "breast":
        data = pd.read_csv(
            "data/wdbc.data", header=None,
        )

        data[data == "?"] = np.nan
        data.dropna(axis=0, inplace=True)
        y_data = data.iloc[:, 1].values  # convert strings to integer
        x_data = data.iloc[:, 2:,].values

        # set to binary
        y_data[y_data == "B"] = 0  # benign
        y_data[y_data == "M"] = 1  # malignant
        y_data = y_data.astype("int")

    elif args.data_name == "ionosphere":
        data = pd.read_csv(
            "data/ionosphere_class.data",
            header=None,
        )
        y_data = data.iloc[:, 34].values  # convert strings to integer
        x_data = data.iloc[:, 0:34]
        x_data = x_data.drop(1, axis=1).values  # drop constant columns

        # set to binary
        y_data[y_data == "g"] = 1  # good
        y_data[y_data == "b"] = 0  # bad
        y_data = y_data.astype("int")

    elif args.data_name == "parkinsons":
        data = pd.read_csv(
            "data/parkinsons_class.data"
        )
        data[data == "?"] = np.nan
        data.dropna(axis=0, inplace=True)
        y_data = data["status"].values  # convert strings to integer
        x_data = data.drop(columns=["name", "status"]).values

    elif args.data_name == "mnist":
        # y, lab = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        # k = ckern.Conv(GPflow.kernels.RBF(25, ARD=self.run_settings['kernel_ard']), [28, 28],
        #                [5, 5]) + GPflow.kernels.White(1, 1e-3)
        from torchvision.datasets import MNIST

        train_data = MNIST(args.data_path, train=True, download=True)
        x_data = train_data.data.numpy().reshape((-1, 784))
        y_data = train_data.targets.numpy()
        y_data, x_data = y_data[y_data < 2], x_data[y_data < 2]
        
        trn_idx, val_idx = train_test_split(
            np.arange(len(y_data)), test_size=0.05, random_state=args.seed + 3
        )
        y_val = np.concatenate((x_data[val_idx], y_data[val_idx][:, None]), axis=1)

        x_data = x_data[trn_idx]
        y_data = y_data[trn_idx]
        test_data = MNIST(args.data_path, train=False, download=True)
        x_test = test_data.data.numpy().reshape((-1, 784))
        y_test = test_data.targets.numpy()
        y_test = np.concatenate(
            (x_test[y_test < 2], y_test[y_test < 2][:, None]), axis=1
        )
        y = np.concatenate((x_data, y_data[:, None]), axis=1)
        to_drop = np.where(np.std(y, axis=0) == 0)[0]
        y = np.delete(y, to_drop, 1)
        y_val = np.delete(y_val, to_drop, 1)
        y_test = np.delete(y_test, to_drop, 1)

        norm_idx = y.shape[1] - 1
        y[:, :norm_idx] = y[:, :norm_idx] + np.random.uniform(
            0, 1, size=y[:, :norm_idx].shape
        )
        y_val[:, :norm_idx] = y_val[:, :norm_idx] + np.random.uniform(
            0, 1, size=y_val[:, :norm_idx].shape
        )
        y_test[:, :norm_idx] = y_test[:, :norm_idx] + np.random.uniform(
            0, 1, size=y_test[:, :norm_idx].shape
        )
        y = y.astype("float")
        y_val = y_val.astype("float")
        y_test = y_test.astype("float")
        y[:, :norm_idx] = y[:, :norm_idx] / 256.0
        y_val[:, :norm_idx] = y_val[:, :norm_idx] / 256.0
        y_test[:, :norm_idx] = y_test[:, :norm_idx] / 256.0
        
        def logit(x):
            return np.log(x / (1.0 - x))

        def logit_transform(x):
            return logit(1e-10 + (1 - 2e-10) * x)

        y[:, :norm_idx] = logit_transform(y[:, :norm_idx])
        y_val[:, :norm_idx] = logit_transform(y_val[:, :norm_idx])
        y_test[:, :norm_idx] = logit_transform(y_test[:, :norm_idx])

        x_train = y[:, :norm_idx]
        y_train = y[:, norm_idx].astype("int")
        x_test = y_test[:, :norm_idx]
        y_test = y_test[:, norm_idx].astype("int")

    else:
        print("Dataset doesn't exist")
        raise NotImplementedError


elif args.regress:
    if args.data_name == "concrete":
        data = pd.read_excel(
            "data/Concrete_Data.xls"
        )
        y_data = data.iloc[:, 8].values
        x_data = data.iloc[:, 0:8].values
    elif args.data_name == "wine":
        data = pd.read_csv(
            "data/winequality-red.csv", sep=";"
        )
        y_data = data.iloc[:, 11].values  # convert strings to integer
        x_data = data.iloc[:, 0:11].values
    elif args.data_name == "boston":
        from sklearn.datasets import load_boston

        x_data, y_data = load_boston(return_X_y=True)
    elif args.data_name == "diabetes":
        from sklearn.datasets import load_diabetes

        x_data, y_data = load_diabetes(return_X_y=True)
    else:
        print("Dataset doesn't exist")
        raise NotImplementedError
    
if args.data_name != 'mnist':
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.5, random_state=args.seed
    )

# scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
args.test_batch_size = min(args.batch_size, x_test.shape[0])
args.batch_size = min(args.batch_size, x_train.shape[0])

if args.regress:
    scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train = scaler.transform(y_train.reshape(-1, 1))
    y_test = scaler.transform(y_test.reshape(-1, 1))

trainloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float() if args.regress else torch.from_numpy(y_train)),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
)

testloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_train).float() if args.regress else torch.from_numpy(y_test)),
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=0,
)

        
# Model
print('==> Building model..')
class MLP(torch.nn.Module):
    def __init__(self, channels):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(channels)-1):
            layers.append(nn.Linear(channels[i], channels[i+1]))
            if i < len(channels)-2:
                layers.append(nn.ReLU())
            # elif args.regress == 0:
            #     layers.append(nn.ReLU())
                
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
        

net = MLP([x_train.shape[-1], args.num_hidden, 2])
use_cuda = False
if use_cuda:
    net.cuda(device_id)
    cudnn.benchmark = True
    cudnn.deterministic = True

# print(net(torch.from_numpy(x_train[:5])).float())

def update_params(lr,epoch):
    for p in net.parameters():
        if not hasattr(p,'buf'):
            if use_cuda:
                p.buf = torch.zeros(p.size()).cuda(device_id)
            else:
                p.buf = torch.zeros(p.size())
        d_p = p.grad.data
        d_p.add_(weight_decay, p.data)
        buf_new = (1-args.alpha)*p.buf - lr*d_p
        if (epoch%args.cycle_length)+1>args.cycle_length - args.offset:
            if use_cuda:
                eps = torch.randn(p.size()).cuda(device_id)
            else:
                eps = torch.randn(p.size())
            buf_new += (2.0*lr*args.alpha*args.temperature/(2*datasize))**.5*eps
        p.data.add_(buf_new)
        p.buf = buf_new

def adjust_learning_rate(epoch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0
    return lr

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
        net.zero_grad()
        lr = adjust_learning_rate(epoch,batch_idx)
        outputs = net(inputs)
        if args.regress == 0:
            loss = criterion(outputs, targets)
        else:
            loss = criterion(outputs[:, 0], targets, outputs[:, 1].exp())
        loss.backward()
        update_params(lr, epoch)

        train_loss += loss.data.item()
        
        if args.regress == 0:
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            
    if epoch%args.cycle_length==0:
        print('\nEpoch: %d' % epoch)
        if args.regress == 0:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))
        else:
            print('Loss: %.3f' % (train_loss/(batch_idx+1)))
            
    wandb.log({'train_loss': train_loss/(batch_idx+1)})
        
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            if args.regress == 0:
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

    if epoch%10==0:
        if args.regress == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss/len(testloader), correct, total,
                100. * correct.item() / total))
        else:
            print('\nTest set: Average loss: {:.4f}\n'.format(
                test_loss/len(testloader)))
    wandb.log({"test_loss": test_loss/len(testloader)})
        
        
weight_decay = args.weight_decay
datasize = x_train.shape[0]
num_batch = datasize/args.batch_size+1
lr_0 = args.lr_0 # initial lr
M = args.epochs // args.cycle_length # number of cycles
T = args.epochs*num_batch # total number of iterations
if args.regress == 0:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.GaussianNLLLoss()
    
mt = 0

if args.regress == 0:
    def predict_density(inputs, all_preds):
        inputs = inputs.float()
        logp = torch.log(torch.nn.functional.softmax(all_preds.mean(1), dim=1))
        return inputs * logp[:, 1] + (1 - inputs) * logp[:, 0]
else:
    def predict_density(inputs, all_preds):
        mean, std = all_preds[..., 0].mean(1), all_preds[..., 1].exp() ** .5
        return sp.stats.norm.logpdf(
            inputs[:, -1], loc=mean, scale=std
        )

print('==> Starting training..')
all_samples = torch.zeros((x_test.shape[0], args.epochs // args.cycle_length, 1 + (args.regress==0)))
for epoch in range(args.epochs):
    train(epoch)
    if epoch%args.cycle_length==0:
        test(epoch)
    if (epoch%args.cycle_length)+1>args.cycle_length-1:
        print('save!')
        if use_cuda:
            net.cpu()
        torch.save(net.state_dict(),args.dir + '/%s_%i_csghmc_%i.pt'%(args.data_name, args.seed, mt))
        if use_cuda:
            net.cuda(device_id)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                all_samples[:, mt] = net(inputs)
                if batch_idx > 0:
                    raise ValueError('batch size too small')
        mt += 1

nll = -predict_density(targets, all_samples).mean()
print('nll', nll.item())

wandb.log({'nll': nll.item()})