
# coding: utf-8

# In[192]:


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

from PIL import Image


# In[193]:


def PCA(data, k=30):
    X = data
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)

    # svd
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])


# In[194]:


def generate_gmm(samples, k=5):
    gmm = GaussianMixture(n_components = k, max_iter=1000)
    gmm = gmm.fit(samples)
    covars = gmm.covariances_
    means = gmm.means_
    weights = gmm.weights_
    return gmm, weights, means, covars


# In[195]:


def likelihood_moment(x, ytk, moment):
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk


# In[196]:


def likelihood_statistics(samples, weights, means, covars):
    normals = [multivariate_normal(mean=means[k], cov=covars[k]) for k in range(0, len(weights))]

    gaussian_pdfs = np.transpose(np.array(list(g_k.pdf(samples) for g_k in normals)))
    
    s0,s1,s2 = {}, {}, {}
    
    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in enumerate(samples):
            probabilities = np.multiply(gaussian_pdfs[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)
    
    return s0, s1, s2


# In[197]:


def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.linalg.norm(v,axis = 1)[:,:,None,None]



# In[198]:


def fisher_vector(samples, weights, means, covars):
    s0, s1, s2 =  likelihood_statistics(samples, weights, means, covars)
    #print(s1)
    T = samples.shape[0]
    #print(T)
    covs = np.float32([np.diagonal(covars[k]) for k in range(0, covars.shape[0])])
    a = fisher_vector_means(s0, s1, s2, means, covs, weights, T)
    b = fisher_vector_sigma(s0, s1, s2, means, covs, weights, T)
    fv = np.concatenate([np.concatenate(a), np.concatenate(b)])
    fv = normalize(fv)
    return fv



