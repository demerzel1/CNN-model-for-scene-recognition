
# coding: utf-8

# In[1]:


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


# In[2]:


def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


# In[3]:


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant(m.bias.data,0.1)     


# In[4]:


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.005 * (0.2 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# In[5]:


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


# In[6]:


def train_model(model, criterion1, criterion2, optimizer, scheduler, num_epochs, use_gpu):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        begin_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #adjust_learning_rate(optimizer,epoch)
        
        # Each epoch has a training and validation phase
        for phase in ['Scene15-train1', 'Scene15-test1']:
            count_batch = 0
            if phase == 'Scene15-train1':
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
                lcs_output,outputs = model(inputs)
                _1, preds1 = torch.max(outputs.data, 1)
                _2, preds2 = torch.max(lcs_output.data, 1)
                loss1 = criterion1(outputs, labels)
                loss2 = criterion2(lcs_output, labels)

                loss = loss1+0.2*loss2
                
                # backward + optimize only if in training phase
                if phase == 'Scene15-train1':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects1 += torch.sum(preds1 == labels.data).to(torch.float32)
                running_corrects2 += torch.sum(preds2 == labels.data).to(torch.float32)
                
                # print result every 10 batch
                if count_batch%10 == 0:
                    batch_loss = running_loss / (batch_size*count_batch)
                    batch_acc1 = running_corrects1 / (batch_size*count_batch)
                    batch_acc2 = running_corrects2 / (batch_size*count_batch)
                    print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Time: {:.4f}s'.                           format(phase, epoch, count_batch, batch_loss, batch_acc1, batch_acc2, time.time()-begin_time))
                    begin_time = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc1 = running_corrects1 / dataset_sizes[phase]
            epoch_acc2 = running_corrects2 / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f}'.format(phase, epoch_loss, epoch_acc1, epoch_acc2))

            # save model
            #if phase == 'Scene15-train':
                #if not os.path.exists('output'):
                    #os.makedirs('output')
                #torch.save(model, 'output/15-resnet_epoch{}.pkl'.format(epoch))

            # deep copy the model
            if phase == 'Scene15-test1' and epoch_acc1 > best_acc:
                best_acc = epoch_acc1
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[7]:


class AlexNet(nn.Module):

    def __init__(self, num_classes=15):
        super(AlexNet, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.LCS = nn.Sequential(
            nn.Conv2d(256, 80, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.LCS_classifier = nn.Linear(80 * 6 * 6, num_classes)
        self.features2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features1(x)
        lcs = self.LCS(x)
        lcs = lcs.view(lcs.size(0), 80 * 6 * 6)
        lcs = self.LCS_classifier(lcs)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc(x)
        x = self.classifier(x)
        return lcs,x


# In[8]:


if __name__ == '__main__':
    
    data_transforms = {
        'Scene15-train1': transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Scene15-test1': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    use_gpu = torch.cuda.is_available()
    
    batch_size = 32
    num_class = 15

    image_datasets = {x: customData(img_path='./15-Scene/',
                                    txt_path=('./' + x + '.txt'),
                                    data_transforms = data_transforms,
                                    dataset=x) for x in ['Scene15-train1', 'Scene15-test1']}

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True) for x in ['Scene15-train1', 'Scene15-test1']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['Scene15-train1', 'Scene15-test1']}

    print(dataset_sizes)
    

     # get model and replace the original fc layer with your fc layer
    model_ft = AlexNet()
    #model_ft = models.alexnet(pretrained=False)
    #num_ftrs = model_ft.classifier[-1].in_features
    #model_ft.classifier[-1] = nn.Linear(num_ftrs, num_class)
    #model_ft = models.resnet18(pretrained=False)
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, num_class)

    model_ft.apply(weights_init)
    
    arch = 'alexnet'
    model_file = '%s_places365.pth.tar' % arch
    
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    dict_trained = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    dict_new = model_ft.state_dict().copy()

    new_list = list (model_ft.state_dict().keys() )
    trained_list = list (dict_trained.keys()  )
    print("new_state_dict size: {}  trained state_dict size: {}".format(len(new_list),len(trained_list)) )
    print("New state_dict first 10th parameters names")
    print(new_list[:])
    print("trained state_dict first 10th parameters names")
    print(trained_list[:])

    for i in range(8):
        dict_new[ new_list[i] ] = dict_trained[ trained_list[i] ]
        
    for i in range(6):
        ind1 = 12 + i
        ind2 = 8 + i
        dict_new[ new_list[ind1] ] = dict_trained[ trained_list[ind2] ]
        #print(new_list[ind1])
        #print(trained_list[ind2])


    model_ft.load_state_dict(dict_new)

    # if use gpu
    if use_gpu:
        model_ft = model_ft.cuda()

    # define cost function
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.2)

    # multi-GPU
    model_ft = torch.nn.DataParallel(model_ft, device_ids=[0,1])

    # train model
    model_ft = train_model(model=model_ft,
                           criterion1=criterion1,
                           criterion2=criterion2,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=80,
                           use_gpu=use_gpu)

    # save best model
    #torch.save(model_ft,"output/15-best_alex_lcs_place_preall.pkl")
    torch.save(model_ft.state_dict(), 'output/15-best_alex_lcs_place_params_preall.pkl')  

