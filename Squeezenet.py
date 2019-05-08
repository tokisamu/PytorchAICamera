import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy
import onnx
import onnx_caffe2.backend
from onnx_caffe2.backend import Caffe2Backend
import train
#get image and normalize it
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
print(class_names)
#preview data set
imgs, classes = next(iter(dataloaders['train']))
preview = torchvision.utils.make_grid(imgs)
preview = preview/2+0.5 #unnormalize
npimg = preview.numpy()
plt.imshow(numpy.transpose(npimg,(1,2,0)))
#plt.show()
#fine-tunning a squeezenet
model_squ = models.squeezenet1_1(pretrained=True)
model_squ.features._modules["2"] = torch.nn.MaxPool2d(kernel_size=3, stride=2, dilation=1,ceil_mode=False) #change output layer and a mistake
model_squ.classifier._modules["1"]=torch.nn.Conv2d(512, 6, kernel_size=(1, 1), stride=(1, 1)) #we have six classes
model_squ.num_classes=6 # this is a variable must be changed, otherwise the net doesn't work
print(model_squ)
#define the loss function
e = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(model_squ.parameters(), lr=0.001, momentum=0.92)
exp = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.1)
#start train
training = train.train_model
model_squ = training(model_squ, e, opt, exp, num_epochs=20)
#transfer thie model to a caffe2 model
x = Variable(torch.randn(1,3,224,224), requires_grad=True)
onnxModel = torch.onnx._export(model_squ,x,"sqz.onnx",export_params=True)
model = onnx.load("sqz.onnx")
prepared_backend = onnx_caffe2.backend.prepare(model)
init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(model.graph)
with open("squeeze_init_net.pb", "wb") as f:
    f.write(init_net.SerializeToString())
with open("squeeze_predict_net.pb", "wb") as f:
    f.write(predict_net.SerializeToString())