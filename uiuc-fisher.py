
# coding: utf-8

# In[20]:


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
from sklearn import svm
from sklearn.linear_model import PassiveAggressiveClassifier
import fisher
from sklearn.externals import joblib

from PIL import Image


# In[21]:


def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


# In[22]:


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
        print(classes)
        print(class_to_idx)
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


# In[27]:


def train_model(model, model_2, criterion1, criterion2, num_epochs, use_gpu):
    print("train")
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    begin_time = time.time()
    
    phase = 'uiuc-train'
    clf1 = svm.SVC()
    #clf = SGDClassifier()
    input_svc = np.zeros([dataset_sizes[phase], 4096 * 4])
    label_svc = np.empty(0)
    input_lcs = np.zeros([dataset_sizes[phase], 169, 256])
    input_lcs_2 = np.zeros([dataset_sizes[phase], 169, 256])
    input_fc = np.zeros([dataset_sizes[phase], 4096* 2])
    cnt_fc = 0
    cnt_lcs = 0
    cnt_lcs_2 = 0
    count_batch = 0
    model.eval()
    model.train(False)  # Set model to evaluate mode

    model_2.eval()
    model_2.train(False)

    running_loss = 0.0
    running_corrects1 = 0.0
    running_corrects2 = 0.0
    svm_corrects = 0.0

    running_loss_2 = 0.0
    running_corrects1_2 = 0.0
    running_corrects2_2 = 0.0

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

        # forward
        fmap_conv4, feature_fc, lcs_output, outputs = model(inputs)
        fmap_conv4_2, feature_fc_2, lcs_output_2, outputs_2 = model_2(inputs)
           
        

        #print(count_batch)
        #[32,80,36]
        fmap_conv4 = fmap_conv4.view(fmap_conv4.size(0), 256, 169)
        #print(fmap_conv4.shape)
        fmap_conv4_2 = fmap_conv4_2.view(fmap_conv4.size(0), 256, 169)
        
        
        #[32,36,80]
        fmap_conv4 = fmap_conv4.permute(0,2,1)
        fmap_conv4_2 = fmap_conv4_2.permute(0,2,1)
        #print(fmap_conv4.shape)
        fmap_conv4 = fmap_conv4.cpu().detach().numpy()
        fmap_conv4_2 = fmap_conv4_2.cpu().detach().numpy()
        
        for index, x in enumerate(fmap_conv4):
            input_lcs[cnt_lcs] = x
            cnt_lcs += 1
            
        for index, x in enumerate(fmap_conv4_2):
            input_lcs_2[cnt_lcs_2] = x
            cnt_lcs_2 += 1

        feature_fc = feature_fc.cpu().detach().numpy()
        feature_fc_2 = feature_fc_2.cpu().detach().numpy()

        labels_svm = labels_svm.detach().numpy()

        for index, x in enumerate(feature_fc):
            input_fc[cnt_fc] = np.concatenate([x, feature_fc_2[index]])
            #input_fc[cnt_fc] = x
            cnt_fc += 1
        label_svc = np.concatenate([label_svc,labels_svm])


        #print(clf.predict(f_all))
        #print("{} score".format(clf.score(f_all, labels_svm)))

        #svm_pred = clf.predict(f_all)

        #svm_pred = torch.from_numpy(svm_pred).cuda()
        #svm_corrects += torch.sum(svm_pred == labels.data).to(torch.float32)

        _1, preds1 = torch.max(outputs.data, 1)
        _2, preds2 = torch.max(lcs_output.data, 1)
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(lcs_output, labels)
        loss = loss1+0.5*loss2

        _1_2, preds1_2 = torch.max(outputs_2.data, 1)
        _2_2, preds2_2 = torch.max(lcs_output_2.data, 1)
        loss1_2 = criterion1(outputs_2, labels)
        loss2_2 = criterion2(lcs_output_2, labels)
        loss_2 = loss1_2+0.5*loss2_2

        # statistics
        running_loss += loss.item()
        running_corrects1 += torch.sum(preds1 == labels.data).to(torch.float32)
        running_corrects2 += torch.sum(preds2 == labels.data).to(torch.float32)

        running_loss_2 += loss_2.item()
        running_corrects1_2 += torch.sum(preds1_2 == labels.data).to(torch.float32)
        running_corrects2_2 += torch.sum(preds2_2 == labels.data).to(torch.float32)

        # print result every 10 batch
        if count_batch%10 == 0:
            batch_loss = running_loss / (batch_size*count_batch)
            batch_acc1 = running_corrects1 / (batch_size*count_batch)
            batch_acc2 = running_corrects2 / (batch_size*count_batch)

            batch_loss_2 = running_loss_2 / (batch_size*count_batch)
            batch_acc1_2 = running_corrects1_2 / (batch_size*count_batch)
            batch_acc2_2 = running_corrects2_2 / (batch_size*count_batch)


            #batch_svm_acc = svm_corrects / (batch_size*count_batch)
            print('{} Batch [{}] Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Time: {:.4f}s'.                   format(phase, count_batch, batch_loss, batch_acc1, batch_acc2, time.time()-begin_time))

            print('{} Batch [{}] Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f}  Time: {:.4f}s'.                   format(phase, count_batch, batch_loss_2, batch_acc1_2, batch_acc2_2, time.time()-begin_time))

            begin_time = time.time()

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc1 = running_corrects1 / dataset_sizes[phase]
    epoch_acc2 = running_corrects2 / dataset_sizes[phase]

    epoch_loss_2 = running_loss_2 / dataset_sizes[phase]
    epoch_acc1_2 = running_corrects1_2 / dataset_sizes[phase]
    epoch_acc2_2 = running_corrects2_2 / dataset_sizes[phase]
    print('{} Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} '.format(phase, epoch_loss, epoch_acc1, epoch_acc2))

    print('{} Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} '.format(phase, epoch_loss_2, epoch_acc1_2, epoch_acc2_2))
    
    
    input_pca = torch.zeros(dataset_sizes[phase] * 169, 32)
    cnt_pca = 0
    
    for t in input_lcs:
        #print(t.shape)
        t = torch.Tensor(t)
        t = fisher.PCA(t, 32)
        
        #print(t.shape)
        for tt in t:
            input_pca[cnt_pca] = tt
            cnt_pca += 1
    print("input_pca")        
    print(input_pca.shape)
    
    
    gmm, weights, means, covars = fisher.generate_gmm(input_pca.cpu().detach().numpy(), 64)
    #gmm = joblib.load("gmm_model.m")
    #covars = gmm.covariances_
    #means = gmm.means_
    #weights = gmm.weights_
    #print("gmm ok")
    
    print(weights)
    print(means)
    print(covars)
    joblib.dump(gmm, "gmm_uiuc.m")
    print("gmm ok")
    
    #fv_all = np.zeros([dataset_sizes[phase], 4096])
    cnt = 0
    
    for index,t in enumerate(input_lcs):
        t = torch.Tensor(t)
        t = fisher.PCA(t, 32)
        t = t.cpu().detach().numpy()
        fv = fisher.fisher_vector(t, weights, means, covars)
        
        t_2 = torch.Tensor(input_lcs_2[index])
        t_2 = fisher.PCA(t_2, 32)
        t_2 = t_2.cpu().detach().numpy()
        
        fv_2 = fisher.fisher_vector(t_2, weights, means, covars)
        #print(fv.shape)
        #fv_all[cnt] = fv
        input_svc[cnt] = np.concatenate([input_fc[cnt], fv, fv_2])
        cnt += 1
    
    print(input_svc.shape)
    #print(feature_fc.shape)
    #print(fv_all.shape[1])
    #print(feature_fc.size(1))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    print(label_svc.shape)
    clf1.fit(input_svc,label_svc)
    joblib.dump(clf1, "svmall_uiuc.m")
    
    return gmm, clf1


# In[32]:


def test_model(model, model_2, gmm, criterion1, criterion2, clf, use_gpu):
    
    
    print("test")
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    begin_time = time.time()
    
    phase = 'uiuc-test'
    
    y_true = np.zeros(dataset_sizes[phase])
    y_pred = np.zeros(dataset_sizes[phase])
    cnt_true = 0
    cnt_pred = 0
    
    count_batch = 0
    model.eval()
    model.train(False)  # Set model to evaluate mode

    model_2.eval()
    model_2.train(False)
    
    running_loss = 0.0
    running_corrects1 = 0.0
    running_corrects2 = 0.0
    svm_corrects = 0.0
    
    running_loss_2 = 0.0
    running_corrects1_2 = 0.0
    running_corrects2_2 = 0.0
    
    covars = gmm.covariances_
    means = gmm.means_
    weights = gmm.weights_
    
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
        fmap_conv4, feature_fc, lcs_output, outputs = model(inputs)
        fmap_conv4_2, feature_fc_2, lcs_output_2, outputs_2 = model_2(inputs)
        
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
        fmap_conv4 = fmap_conv4.view(fmap_conv4.size(0), 256, 169)
        fmap_conv4_2 = fmap_conv4_2.view(fmap_conv4.size(0), 256, 169)
        
        #[32,36,80]
        fmap_conv4 = fmap_conv4.permute(0,2,1)
        fmap_conv4_2 = fmap_conv4_2.permute(0,2,1)
        
        fv_all = np.empty([fmap_conv4.size(0),4096 * 2])
            
        cnt = 0
        for index,t in enumerate(fmap_conv4):
            #print(t.shape)
            t = fisher.PCA(t, 32)
            #print(t.shape)
            t_np = t.cpu().detach().numpy()
            
            fv = fisher.fisher_vector(t_np, weights, means, covars)
            
            t_2 = fisher.PCA(fmap_conv4_2[index], 32)
            #print(t.shape)
            t_np_2 = t_2.cpu().detach().numpy()
            
            fv_2 = fisher.fisher_vector(t_np_2, weights, means, covars)
            fv_all[cnt] = np.concatenate([fv, fv_2])
            
            #fv_all[cnt] = t_np.flatten()
            cnt += 1
       
        
        
        #f_all = fv_all
        #print(f_all.shape)
        
        feature_fc = feature_fc.cpu().detach().numpy()
        feature_fc_2 = feature_fc_2.cpu().detach().numpy()
        '''
        f_all = np.empty([feature_fc.shape[0], feature_fc.shape[1] + fv_all.shape[1] ])
        for index, x in enumerate(feature_fc):
            f_all[index] = np.concatenate([x, fv_all[index]])
        '''
        f_all = np.empty([feature_fc.shape[0], feature_fc.shape[1] + feature_fc_2.shape[1] + fv_all.shape[1] ] )
        for index, x in enumerate(feature_fc):
            f_all[index] = np.concatenate([x, feature_fc_2[index], fv_all[index]])
        
        #print(labels_svm)
        labels_svm = labels_svm.detach().numpy()
        #print(clf.predict(f_all))
        #print("{} score".format(clf.score(fv_all, labels_svm)))
        
        svm_pred = clf.predict(f_all).astype(np.int64)
        
        for x in labels_svm:
            y_true[cnt_true] = x
            cnt_true += 1
        for x in svm_pred:
            y_pred[cnt_pred] = x
            cnt_pred +=1
            
        #print(svm_pred)
        #print(svm_pred.dtype)
        #print(labels.data.dtype)
        svm_pred = torch.from_numpy(svm_pred)
        svm_corrects += torch.sum(svm_pred == labels.data).to(torch.float32)

        _1, preds1 = torch.max(outputs.data, 1)
        _2, preds2 = torch.max(lcs_output.data, 1)
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(lcs_output, labels)
        loss = loss1+0.5*loss2
        
        _1_2, preds1_2 = torch.max(outputs_2.data, 1)
        _2_2, preds2_2 = torch.max(lcs_output_2.data, 1)
        loss1_2 = criterion1(outputs_2, labels)
        loss2_2 = criterion2(lcs_output_2, labels)
        loss_2 = loss1_2+0.5*loss2_2

        # statistics
        running_loss += loss.item()
        running_corrects1 += torch.sum(preds1 == labels.data).to(torch.float32)
        running_corrects2 += torch.sum(preds2 == labels.data).to(torch.float32)

        running_loss_2 += loss_2.item()
        running_corrects1_2 += torch.sum(preds1_2 == labels.data).to(torch.float32)
        running_corrects2_2 += torch.sum(preds2_2 == labels.data).to(torch.float32)
        
        # print result every 10 batch
        if count_batch%10 == 0:
            batch_loss = running_loss / (batch_size*count_batch)
            batch_acc1 = running_corrects1 / (batch_size*count_batch)
            batch_acc2 = running_corrects2 / (batch_size*count_batch)
            
            batch_loss_2 = running_loss_2 / (batch_size*count_batch)
            batch_acc1_2 = running_corrects1_2 / (batch_size*count_batch)
            batch_acc2_2 = running_corrects2_2 / (batch_size*count_batch)
            
            batch_svm_acc = svm_corrects / (batch_size*count_batch)
            print('{} Batch [{}] Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Svm Acc: {:.4f} Time: {:.4f}s'.                   format(phase, count_batch, batch_loss, batch_acc1, batch_acc2, batch_svm_acc, time.time()-begin_time))

            print('{} Batch [{}] Loss: {:.4f} Acc1_2: {:.4f} Acc2_2: {:.4f} Svm Acc: {:.4f} Time: {:.4f}s'.                   format(phase, count_batch, batch_loss_2, batch_acc1_2, batch_acc2_2, batch_svm_acc, time.time()-begin_time))

            begin_time = time.time()

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc1 = running_corrects1 / dataset_sizes[phase]
    epoch_acc2 = running_corrects2 / dataset_sizes[phase]
    
    epoch_loss_2 = running_loss_2 / dataset_sizes[phase]
    epoch_acc1_2 = running_corrects1_2 / dataset_sizes[phase]
    epoch_acc2_2 = running_corrects2_2 / dataset_sizes[phase]
    
    epoch_svm_acc = svm_corrects / dataset_sizes[phase]
    print('{} Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Svm Acc {:.4f}'.format(phase, epoch_loss, epoch_acc1, epoch_acc2, epoch_svm_acc))

    print('{} Loss_2: {:.4f} Acc1_2: {:.4f} Acc2_2: {:.4f} Svm Acc {:.4f}'.format(phase, epoch_loss_2, epoch_acc1_2, epoch_acc2_2, epoch_svm_acc))

    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model, y_true, y_pred


# In[33]:


class AlexNet(nn.Module):

    def __init__(self, num_classes=8):
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
        lcs = lcs.view(lcs.size(0), 80 * 6 * 6)
        lcs = self.LCS_classifier(lcs)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc(x)
        feature_fc = x
        x = self.classifier(x)
        return fmap_conv4, feature_fc, lcs, x


# In[34]:


if __name__ == '__main__':
    
    #output=sys.stdout
    #outputfile=open("alexnet_BN.txt","w")
    #sys.stdout=outputfile
    
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
    model = AlexNet()
    
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    
    model.load_state_dict(torch.load('output/uiuc-lcs-image-params.pkl'))
    
    model_2 = AlexNet()
    
    model_2 = torch.nn.DataParallel(model_2, device_ids=[0,1])
    
    model_2.load_state_dict(torch.load('output/uiuc_places_params.pkl'))
    
    # if use gpu
    if use_gpu:
        model = model.cuda()
        model_2 = model_2.cuda()

    # define cost function
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()


    if os.path.exists('svmall_uiuc.m'):
        clf = joblib.load("svmall_uiuc.m")
        gmm = joblib.load("gmm_uiuc.m")
    else:
        gmm, clf = train_model(model=model,
                               model_2 = model_2,
                               criterion1=criterion1,
                               criterion2=criterion2,
                               num_epochs = 1,
                               use_gpu=use_gpu)
    #test
    model, y_true, y_pred = test_model(model=model,
                           model_2 = model_2,
                           gmm = gmm,
                           criterion1=criterion1,
                           criterion2=criterion2,
                           clf = clf,
                           use_gpu=use_gpu)

    # save best model
    #torch.save(model,"output/15-best_alex_lcs.pkl")


# In[35]:


image_datasets = {x: customData(img_path='../data/',
                                    txt_path=('../' + x + '.txt'),
                                    data_transforms = data_transforms,
                                    dataset=x) for x in ['uiuc-train', 'uiuc-test']}

    # wrap your data and label into Tensor
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True) for x in ['uiuc-train', 'uiuc-test']}


# In[36]:


import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# In[37]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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



# In[38]:



np.set_printoptions(precision=2)

class_names = ['RockClimbing', 'Rowing', 'badminton', 'bocce', 'croquet', 'polo', 'sailing', 'snowboarding']
#y_true = y_true.tolist()
#y_pred = y_pred.tolist()

# Plot non-normalized confusion matrix
ax,cm = plot_confusion_matrix(y_true, y_pred, classes=class_names,
                      title='Confusion matrix')

plt.show()


# In[41]:


for i in range(8):
    print(cm[i][i]/cm[i].sum())


# In[42]:


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

