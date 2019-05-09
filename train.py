
#Imports Here
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import json
from collections import OrderedDict 
import timeit


import argparse


parser = argparse.ArgumentParser(description='This program tests and trains the network and saves a checkpoint ')
parser.add_argument('--usegpu', dest="gpu", action="store", default="cpu")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=1.0)
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=20)
parser.add_argument('--pmodel', dest="pmodel", action="store", default="alexnet", type = str)

args = parser.parse_args()




#dropout load from argparse
dropout = args.dropout


#device Select
gpu = args.gpu
device = torch.device("cuda:0" if torch.cuda.is_available() and gpu=="gpu"  else "cpu")
print ("Training using {}".format(device))
#epoch from argparse
epochs =args.epochs


#learning rate from argparse
learning_rate = args.learning_rate
learningRate = learning_rate
print("with learning rate  {}".format(learningRate))

#model and feature load model argparse 
pmodel = args.pmodel
if pmodel == 'vgg16':
    model = models.vgg16(pretrained=True)
    feature=25088
elif pmodel == 'alexnet':
    model = models.alexnet(pretrained=True)
    feature=9216
else:
    print("Im sorry but {} is not a valid model.Did you mean vgg16 or alexnet?".format(structure))
print("using {}".format(pmodel))
    
#directory structure for project dataset 


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#transforms for the training, validation, and testing sets


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


#Define data transforms
data_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(norm_mean,
                                                             norm_std)])
#Define training data transforms
data_trans_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_mean,
                                                          norm_std)])
#Load the datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_trans_train)
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms)
    
#Define dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    
class_idx = train_dataset.class_to_idx




#label mapping



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
 


 
 #define network


classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(feature, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))    
   
for param in model.parameters():
    param.requires_grad = False
        
    
        
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adadelta(model.classifier.parameters(), learningRate )
model.cuda()

# TODO: Do validation on the test set
start = timeit.default_timer()

print_every = 5
steps = 0
loss_show=[]

model.to(device)
for e in range(epochs):
    
    running_loss = 0
    for i, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        inputs,labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            vlost = 0
            accuracy=0
            
            
            for i, (inputs2,labels2) in enumerate(validloader):
                optimizer.zero_grad()
                
                inputs2, labels2 = inputs2.to(device) , labels2.to(device)
                model.to(device)
                with torch.no_grad():    
                    outputs = model.forward(inputs2)
                    vlost = criterion(outputs,labels2)
                    ps = torch.exp(outputs).data
                    equality = (labels2.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
            vlost = vlost / len(validloader)
            accuracy = accuracy /len(validloader)
            
                    
            
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Lost {:.4f}".format(vlost),
                   "Accuracy: {:.4f}".format(accuracy))
            
            
            running_loss = 0
            
stop = timeit.default_timer()
print('Time taken for training is :{0:.2f} mins '.format((stop - start)/60))  
def checkAccuracy(testloader):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    a=(100 * correct / total)

    print('Accuracy of the network is: {0:.2f} % ' .format(a))
    
    
checkAccuracy(testloader)

model.class_to_idx = train_dataset.class_to_idx
model.cpu()
torch.save({'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx}, 'check1.pth')
    