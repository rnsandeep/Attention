import os
import csv
import random
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.utils as utils
from torchvision import datasets, models, transforms
from networks import AttnVGG, VGG
from loss import FocalLoss
from data import preprocess_data_2016, preprocess_data_2017, ISIC
from utilities import *
import time, copy
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:#, 'val', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            count = 0
            start = time.time()
            all_times = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if inputs.size()[0] ==1:
                    continue

                count = count +1
                # zero the parameter gradients
                optimizer.zero_grad()
                cls_nums = [torch.sum(labels==i) for i in range(len(class_names))]
                w = torch.FloatTensor(cls_nums)
                w = w/torch.sum(w)
                class_weights = 1/torch.FloatTensor(w).to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                       outputs, _, _ = model(inputs)
                    else:
                       outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
#                    print(preds, labels.data)
                    loss = criterion(outputs, labels)
#                    loss = nn.CrossEntropyLoss()(outputs, labels)
#                    loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, labels)
#                    loss = criterion(outputs, labels, weight=class_weights)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                sample_time = time.time()-start
                start = time.time()
                all_times.append(sample_time)
                sys.stdout.write('count: {:d}/{:d}, average time:{:f} \r' \
                             .format(count*BATCH_SIZE, len(dataloaders[phase])*BATCH_SIZE, np.mean(np.array(all_times))/BATCH_SIZE ))
                sys.stdout.flush()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        save_checkpoint({
            'epoch': epoch + 1,
            #'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': epoch_acc,
            'optimizer' : optimizer_ft.state_dict(),
        }, False, os.path.join(output_dir, str(epoch)+'_checkpoint.pth.tar'))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_acc, epoch


def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def datatransforms(mean_file, crop_size):
#`    mean_std = np.load(mean_file)
    mean = (0.7012, 0.5517, 0.4875) #list(mean_std[0].data.cpu().numpy())
    std = (0.0942, 0.1331, 0.1521) #list(mean_std[1].data.cpu().numpy())
    print("mean and standard deviation:",mean,std)
    data_transforms = {
      'train': transforms.Compose([
        transforms.RandomResizedCrop(224), #crop_size),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize( mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05] ) #mean, std) #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05]) #mean, std)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'test': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05]) #mean, std)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),

    }
    return data_transforms


data_dir = sys.argv[1]
output_dir = sys.argv[2]

mean_file =  sys.argv[3]
crop_size = int(sys.argv[4])

data_transforms = datatransforms( mean_file, crop_size)

if not  os.path.exists(output_dir):
    os.makedirs(output_dir)

BATCH_SIZE=32

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']} #, 'val']}

#weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))
#weights = torch.DoubleTensor(weights)
#sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
              for x in ['train']}                                             

num_classes = int(sys.argv[5]) # no of classes
no_of_epochs = int(sys.argv[6]) # no of epochs to train

net = AttnVGG(num_classes=num_classes, attention=True, normalize_attn=True)
#net = VGG(num_classes=num_classes)
net = net.to(device)

model = net # nn.DataParallel(net, device_ids=device_ids).to(device)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes


# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

lr_lambda = lambda epoch : np.power(0.1, epoch//10)
scheduler = lr_scheduler.LambdaLR(optimizer_ft, lr_lambda=lr_lambda)

#print('use focal loss ...')
criterion = FocalLoss(gama=2., size_average=True, weight=None)

# move to GPU

model_ft, best_acc, epoch = train_model(model, criterion, optimizer_ft, scheduler,
                       num_epochs=no_of_epochs)

