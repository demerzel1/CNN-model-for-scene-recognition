
# coding: utf-8

# In[1]:


import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import cv2 as cv
import time
import os
from torch.utils.data import Dataset
import numpy as np
import pylab as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
import fisher
from sklearn.externals import joblib

from PIL import Image


# In[2]:


def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


# In[3]:


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


# In[4]:


def train_model(model, criterion1, criterion2, optimizer, scheduler, num_epochs, use_gpu):
    print("train")
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    begin_time = time.time()
    
    phase = 'Scene15-train1'
    clf1 = SVC()
    clf = PassiveAggressiveClassifier()
    for epoch in range(num_epochs):
        input_svc = np.zeros([dataset_sizes[phase], 4096])
        label_svc = np.empty(0)
        cnt_svc = 0
        count_batch = 0
        model.eval()
        model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects1 = 0.0
        running_corrects2 = 0.0
        svm_corrects = 0.0


        for data in dataloders[phase]:
            count_batch += 1
            # get the inputs
            inputs, labels = data

            labels_svm = labels

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            fmap_conv4, fmap_lcs, feature_fc, lcs_output, outputs = model(inputs)
            
            
            #print(count_batch)
            #[32,80,36]
            fmap_lcs = fmap_lcs.view(fmap_lcs.size(0), 80, 36)

            #[32,36,80]
            fmap_lcs = fmap_lcs.permute(0,2,1)

            fv_all = np.empty([fmap_lcs.size(0),600])

            cnt = 0
            for t in fmap_lcs:
                #print(t.shape)
                t = fisher.PCA(t, 30)
                #print(t.shape)
                t_np = t.clone().cpu().detach().numpy()
                #gmm
                
                gmm, weights, means, covars = fisher.generate_gmm(t_np, 10)
                fv = fisher.fisher_vector(t_np, weights, means, covars)
                fv_all[cnt] = fv
                
                #fv_all[cnt] = t_np.flatten()
                cnt += 1
            #print(fv_all.shape)
            #print(feature_fc.shape)

            #print(fv_all.shape[1])
            #print(feature_fc.size(1))
            
            f_all = np.empty([feature_fc.size(0),fv_all.shape[1] + feature_fc.size(1)])
            for index, x in enumerate(feature_fc):
                f_all[index] = np.concatenate([fv_all[index], x.cpu().detach().numpy()])

            #print(f_all.shape)
            
            #f_all = fv_all
            
            #f_all = feature_fc.cpu().detach().numpy()
            #print(labels_svm)
            feature_fc = feature_fc.cpu().detach().numpy()
            
            labels_svm = labels_svm.detach().numpy()
            
            for x in feature_fc:
                input_svc[cnt_svc] = x
                cnt_svc += 1
            label_svc = np.concatenate([label_svc,labels_svm])
            
            clf.partial_fit(f_all, labels_svm, classes=np.arange(15))
            
            #print(clf.predict(f_all))
            #print("{} score".format(clf.score(f_all, labels_svm)))

            svm_pred = clf.predict(f_all)
        
            svm_pred = torch.from_numpy(svm_pred).cuda()
            svm_corrects += torch.sum(svm_pred == labels.data).to(torch.float32)

            _1, preds1 = torch.max(outputs.data, 1)
            _2, preds2 = torch.max(lcs_output.data, 1)
            loss1 = criterion1(outputs, labels)
            loss2 = criterion2(lcs_output, labels)
            loss = loss1+0.5*loss2

            # statistics
            running_loss += loss.item()
            running_corrects1 += torch.sum(preds1 == labels.data).to(torch.float32)
            running_corrects2 += torch.sum(preds2 == labels.data).to(torch.float32)

            # print result every 10 batch
            if count_batch%10 == 0:
                batch_loss = running_loss / (batch_size*count_batch)
                batch_acc1 = running_corrects1 / (batch_size*count_batch)
                batch_acc2 = running_corrects2 / (batch_size*count_batch)
                batch_svm_acc = svm_corrects / (batch_size*count_batch)
                print('{} Batch [{}] Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Svm Acc: {:.4f} Time: {:.4f}s'.                       format(phase, count_batch, batch_loss, batch_acc1, batch_acc2, batch_svm_acc, time.time()-begin_time))
                begin_time = time.time()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc1 = running_corrects1 / dataset_sizes[phase]
        epoch_acc2 = running_corrects2 / dataset_sizes[phase]
        epoch_svm_acc = svm_corrects / dataset_sizes[phase]
        print('{} Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Svm Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc1, epoch_acc2, epoch_svm_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    
    print(input_svc.shape)
    print(label_svc.shape)
    clf1.fit(input_svc,label_svc)
    joblib.dump(clf1, "svmfc_model.m")
    
    return clf1


# In[5]:


def test_model(model, criterion1, criterion2, optimizer, scheduler, clf, use_gpu):
    
    print("test")
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    begin_time = time.time()
    
    phase = 'Scene15-test1'
    
    count_batch = 0
    model.eval()
    model.train(False)  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects1 = 0.0
    running_corrects2 = 0.0
    svm_corrects = 0.0
    
    # Iterate over data.
    for data in dataloders[phase]:
        count_batch += 1
        # get the inputs
        inputs, labels = data

        labels_svm = labels

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)


        # zero the parameter gradients
        optimizer.zero_grad()

        '''
        if count_batch == 1:
            print(inputs.shape)
            test_pic = inputs[0].clone().cpu().data.numpy()
            test_pic= 1.0/(1+np.exp(-1*test_pic))
            # to [0,255]
            test_pic=np.round(test_pic*255)
            #test_pic = test_pic.squeeze(0)  # remove the fake batch dimension
            test_pic = test_pic.transpose(1,2,0)
            print(test_pic.shape)
            cv.imwrite('output/img.jpg',test_pic)

        '''
        # forward
        fmap_conv4, fmap_lcs, feature_fc, lcs_output, outputs = model(inputs)

        '''
        fmap_conv4_1 = fmap_conv4[0][0].clone().cpu().data.numpy()
        fmap_conv4_1= 1.0/(1+np.exp(-1*fmap_conv4_1))
        # to [0,255]
        fmap_conv4_1=np.round(fmap_conv4_1*255)
        #test_pic = test_pic.squeeze(0)  # remove the fake batch dimension
        cv.imwrite('output/img_feature.jpg',fmap_conv4_1)
        #print(fmap_conv4_1)
        '''
        
        
        #[32,80,36]
        fmap_lcs = fmap_lcs.view(fmap_lcs.size(0), 80, 36)
    
        
        #[32,36,80]
        fmap_lcs = fmap_lcs.permute(0,2,1)

        fv_all = np.empty([fmap_lcs.size(0),600])

        cnt = 0
        for t in fmap_lcs:
            #print(t.shape)
            t = fisher.PCA(t, 30)
            #print(t.shape)
            t_np = t.clone().cpu().detach().numpy()
            #gmm
            
            gmm, weights, means, covars = fisher.generate_gmm(t_np, 10)
            fv = fisher.fisher_vector(t_np, weights, means, covars)
            fv_all[cnt] = fv
            
            #fv_all[cnt] = t_np.flatten()
            cnt += 1
        
        #print(fv_all.shape)
        #print(feature_fc.shape)

        #print(fv_all.shape[1])
        #print(feature_fc.size(1))
        
        f_all = np.empty([feature_fc.size(0),fv_all.shape[1] + feature_fc.size(1)])
        for index, x in enumerate(feature_fc):
            f_all[index] = np.concatenate([fv_all[index], x.cpu().detach().numpy()])
        
        #f_all = fv_all
        #print(f_all.shape)
        
        feature_fc = feature_fc.cpu().detach().numpy()
        
        #print(labels_svm)
        labels_svm = labels_svm.detach().numpy()
        #print(clf.predict(f_all))
        #print("{} score".format(clf.score(fv_all, labels_svm)))
        
        svm_pred = clf.predict(feature_fc).astype(np.int64)
        #print(svm_pred)
        #print(svm_pred.dtype)
        #print(labels.data.dtype)
        svm_pred = torch.from_numpy(svm_pred).cuda()
        svm_corrects += torch.sum(svm_pred == labels.data).to(torch.float32)

        _1, preds1 = torch.max(outputs.data, 1)
        _2, preds2 = torch.max(lcs_output.data, 1)
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(lcs_output, labels)
        loss = loss1+0.5*loss2

        # statistics
        running_loss += loss.item()
        running_corrects1 += torch.sum(preds1 == labels.data).to(torch.float32)
        running_corrects2 += torch.sum(preds2 == labels.data).to(torch.float32)

        # print result every 10 batch
        if count_batch%10 == 0:
            batch_loss = running_loss / (batch_size*count_batch)
            batch_acc1 = running_corrects1 / (batch_size*count_batch)
            batch_acc2 = running_corrects2 / (batch_size*count_batch)
            batch_svm_acc = svm_corrects / (batch_size*count_batch)
            print('{} Batch [{}] Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Svm Acc: {:.4f} Time: {:.4f}s'.                   format(phase, count_batch, batch_loss, batch_acc1, batch_acc2, batch_svm_acc, time.time()-begin_time))
            begin_time = time.time()

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc1 = running_corrects1 / dataset_sizes[phase]
    epoch_acc2 = running_corrects2 / dataset_sizes[phase]
    epoch_svm_acc = svm_corrects / dataset_sizes[phase]
    print('{} Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Svm Acc {:.4f}'.format(phase, epoch_loss, epoch_acc1, epoch_acc2, epoch_svm_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


# In[6]:


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
        fmap_conv4 = x
        lcs = self.LCS(x)
        fmap_lcs = lcs
        lcs = lcs.view(lcs.size(0), 80 * 6 * 6)
        lcs = self.LCS_classifier(lcs)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc(x)
        feature_fc = x
        x = self.classifier(x)
        return fmap_conv4, fmap_lcs, feature_fc, lcs, x


# In[7]:


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
    model = AlexNet()
    
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    
    model.load_state_dict(torch.load('output/15-best_alex_lcs_params_pre11.pkl'))
    
    # if use gpu
    if use_gpu:
        model = model.cuda()

    # define cost function
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.2)


    if os.path.exists('svm1_model.m'):
        clf = joblib.load("svm1_model.m")
    else:
        clf = train_model(model=model,
                               criterion1=criterion1,
                               criterion2=criterion2,
                               optimizer=optimizer_ft,
                               scheduler=exp_lr_scheduler,
                               num_epochs = 1,
                               use_gpu=use_gpu)
    #test
    model = test_model(model=model,
                           criterion1=criterion1,
                           criterion2=criterion2,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           clf = clf,
                           use_gpu=use_gpu)

    # save best model
    #torch.save(model,"output/15-best_alex_lcs.pkl")

