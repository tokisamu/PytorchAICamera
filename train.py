import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy
# Get image and normalize it
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'varify': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
images = {i: torchvision.datasets.ImageFolder(os.path.join(i), data_transforms[i]) for i in ['train', 'varify']}
dataloaders = {i: torch.utils.data.DataLoader(images[i], batch_size=16, shuffle=True,) for i in ['train', 'varify']}
dataset_sizes = {i: len(images[i]) for i in ['train', 'varify']}
class_names = images['train'].classes
# Define training Principle
def train_model(model, criterion, optimizer, scheduler, num_epochs=1):    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))       
        # Each epoch has a training and validation phase
        for phase in ['train', 'varify']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # 训练
            else:
                model.train(False)  # 测试
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            iter=0
            for data in dataloaders[phase]:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                for i in preds:
                    print(class_names[labels[i]]+' '+class_names[preds[i]])
                loss = criterion(outputs, labels)
                print("phase:%s, epoch:%d/%d  Iter %d: loss=%s"%(phase,epoch,num_epochs-1,iter,str(loss.data.numpy())))
                # backward
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)
                iter += 1
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print('-' * 10)       
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model



