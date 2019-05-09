import torch
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import json
import argparse
parser = argparse.ArgumentParser(description="predicts the class of the image")
parser.add_argument('--image', dest="img", action="store", default="check.jpg",help='file name of the image to predict')
parser.add_argument('--classes', dest="clas", action="store", default="cat_to_name.json",help='file allows user to load alternative json file containing class names')
args=parser.parse_args()

imgpath= args.img


className= args.clas
with open(className, 'r') as f:
    cat_to_name = json.load(f)

    
    
data_dir = 'flowers'
train_dir = data_dir + '/train'

test_dir = data_dir + '/test'
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

data_trans_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_mean,
                                                          norm_std)])
train_dataset = datasets.ImageFolder(train_dir, transform=data_trans_train)




def welcomeBack(filepath):
    checkpoint = torch.load(filepath)
    model = models.alexnet()
        
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    
    return model, checkpoint['class_to_idx']


model,class_to_idx = welcomeBack('check1.pth')
idx_to_class = { v : k for k,v in class_to_idx.items()}


model.class_to_idx = train_dataset.class_to_idx
welcomeBack('check1.pth')
model.eval()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    size = 224
    width, height = image.size
    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)
        
    resized_image = image.resize((width, height))
        
    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    cropped_image = image.crop((x0, y0, x1, y1))
    np_image = np.array(cropped_image) / 255.
    mean = np.array([norm_mean])
    std = np.array([norm_std])     
    np_image_array = (np_image - mean) / std
    np_image_array = np_image.transpose((2, 0, 1))
    
    return np_image_array
    


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.to('cuda:0')
    image = Image.open(image_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)
    var_inputs = Variable(tensor.float().cuda(), volatile=False)
    var_inputs = var_inputs.unsqueeze(0)
    output = model.forward(var_inputs)  
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu()
    classes = ps[1].cpu()
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])
    return probabilities.numpy()[0], mapped_classes

image_path =(imgpath)
probabilities, classes = predict(image_path, model)

print("The probabilities Are")
print(probabilities)
print("The Classes Are")
print(classes)


max_index = np.argmax(probabilities)
max_probability = probabilities[max_index]
label = classes[max_index]
print("Its probably a " + cat_to_name[label])
fig = plt.figure(figsize=(7,7))
ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)

image = Image.open(image_path)
ax1.axis('off')
ax1.set_title(cat_to_name[label])
ax1.imshow(image)
labels = []
for cl in classes:
    labels.append(cat_to_name[cl])
y_pos = np.arange(5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.invert_yaxis()  # probabilities read top-to-bottom
ax2.set_xlabel('Probability')
ax2.barh(y_pos, probabilities, xerr=0, align='center')

plt.show()

