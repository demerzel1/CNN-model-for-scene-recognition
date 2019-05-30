
# coding: utf-8

# In[8]:


import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms

import time
import os
from torch.utils.data import Dataset
from torch.utils import model_zoo

from PIL import Image


# In[9]:


def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


# In[10]:


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)     


# In[11]:


class customData(Dataset):
    
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            
            self.img_name = []
            
            for line in lines:
                if line[-1] == '\n':
                    line = line[:-1]
                self.img_name.append(os.path.join(img_path,line))
            
            classes,class_to_idx = self._find_classes(img_path)
            self.img_label = [class_to_idx[line.strip().split('/')[0]] for line in lines]
            
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader
        
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label


# In[12]:


def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        begin_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['uiuc-train', 'uiuc-test']:
            count_batch = 0
            if phase == 'uiuc-train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects1 = 0.0
            running_corrects2 = 0.0

            # Iterate over data.
            for data in dataloders[phase]:
                count_batch += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _1, preds1 = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                
                # backward + optimize only if in training phase
                if phase == 'uiuc-train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects1 += torch.sum(preds1 == labels.data).to(torch.float32)
                
                # print result every 10 batch
                if count_batch%10 == 0:
                    batch_loss = running_loss / (batch_size*count_batch)
                    batch_acc1 = running_corrects1 / (batch_size*count_batch)
                    print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc1: {:.4f}  Time: {:.4f}s'.                           format(phase, epoch, count_batch, batch_loss, batch_acc1, time.time()-begin_time))
                    begin_time = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc1 = running_corrects1 / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc1: {:.4f}'.format(phase, epoch_loss, epoch_acc1))

            # save model
            #if phase == 'Scene15-train':
                #if not os.path.exists('output'):
                    #os.makedirs('output')
                #torch.save(model, 'output/15-resnet_epoch{}.pkl'.format(epoch))

            # deep copy the model
            if phase == 'uiuc-test' and epoch_acc1 > best_acc:
                best_acc = epoch_acc1
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[13]:


if __name__ == '__main__':
    
    data_transforms = {
        'uiuc-train': transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'uiuc-test': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    use_gpu = torch.cuda.is_available()
    
    batch_size = 32
    num_class = 8

    image_datasets = {x: customData(img_path='../data/',
                                    txt_path=('../' + x + '.txt'),
                                    data_transforms = data_transforms,
                                    dataset=x) for x in ['uiuc-train', 'uiuc-test']}

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True) for x in ['uiuc-train', 'uiuc-test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['uiuc-train', 'uiuc-test']}

    print(dataset_sizes)
    
     # get model and replace the original fc layer with your fc layer
    #model_ft = AlexNet()
    model_ft = models.alexnet(pretrained=False)
    #num_ftrs = model_ft.classifier[-1].in_features
    #model_ft.classifier[-1] = nn.Linear(num_ftrs, num_class)
    #model_ft = models.resnet18(pretrained=False)
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, num_class)
    
    model_ft.apply(weights_init)
    
    # if use gpu
    if use_gpu:
        model_ft = model_ft.cuda()

    # define cost function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.2)

    # multi-GPU
    model_ft = torch.nn.DataParallel(model_ft, device_ids=[0,1])

    # train model
    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=80,
                           use_gpu=use_gpu)

    # save best model
    torch.save(model_ft,"output/uiuc.pkl")


# In[14]:


import numpy as np


# In[17]:


count_batch = 0
phase = 'uiuc-test'

model_ft.train(False)  # Set model to evaluate mode

running_loss = 0.0
running_corrects1 = 0.0
running_corrects2 = 0.0

y_true = np.zeros(dataset_sizes[phase])
y_pred = np.zeros(dataset_sizes[phase])
cnt_true = 0
cnt_pred = 0
# Iterate over data.
for data in dataloders[phase]:
    count_batch += 1
    # get the inputs
    inputs, labels = data

    # wrap them in Variable
    #inputs, labels = Variable(inputs), Variable(labels)

    # forward
    outputs = model_ft(inputs)
    _1, preds1 = torch.max(outputs.data, 1)

    for t in labels:
        y_true[cnt_true] = t
        cnt_true += 1
    for t in preds1:
        y_pred[cnt_pred] = t
        cnt_pred += 1


# In[18]:


import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# In[19]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'
    plt.rcParams['figure.figsize'] = (16.0, 8.0)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm



# In[22]:


np.set_printoptions(precision=2)

class_names = ['RockClimbing', 'Rowing', 'badminton', 'bocce', 'croquet', 'polo', 'sailing', 'snowboarding']


# Plot non-normalized confusion matrix
ax,cm = plot_confusion_matrix(y_true, y_pred, classes=class_names,
                      title='Confusion matrix')

plt.show()


# In[21]:


M = cm
n = len(M)
for i in range(len(M[0])):
    rowsum, colsum = sum(M[i]), sum(M[r][i] for r in range(n))
    try:
        print ('precision: %s' % (M[i][i]/float(colsum)), 'recall: %s' % (M[i][i]/float(rowsum)) )
        p = M[i][i]/float(colsum)
        r = M[i][i]/float(rowsum)
        print('f1: %s' % (2*(p*r)/(p+r)))
    except ZeroDivisionError:
        print ('precision: %s' % 0, 'recall: %s' %0)

