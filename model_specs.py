import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import argparse
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import os
from scipy.stats import norm


class CustomNaiveBayesClassifier:
    def __init__(self, num_classes, boolean_feature_indices, three_cat_feature_indices, continuous_feature_indices, boolean_priors, three_cat_priors, continuous_priors):
        self.num_classes = num_classes
        self.boolean_feature_indices = boolean_feature_indices
        self.three_cat_feature_indices = three_cat_feature_indices
        self.continuous_feature_indices = continuous_feature_indices

        # Parameters for Bernoulli distribution (boolean features)
        self.boolean_probs = torch.zeros(num_classes, len(boolean_feature_indices))
        self.boolean_priors = boolean_priors

        # Parameters for Categorical distribution (three categorical features)
        self.three_cat_prob_1 = torch.zeros(num_classes, len(three_cat_feature_indices))
        self.three_cat_prob_2 = torch.zeros(num_classes, len(three_cat_feature_indices))
        self.three_cat_prob_3 = torch.zeros(num_classes, len(three_cat_feature_indices))
        self.three_cat_priors = three_cat_priors

        # Parameters for Gaussian distribution (continuous features)
        self.mean = torch.zeros(num_classes, len(continuous_feature_indices))
        self.std = torch.ones(num_classes, len(continuous_feature_indices))
        self.continuous_priors =  continuous_priors

    def fit(self, features, labels):
        # Separate boolean and continuous features
        boolean_features = features[:, self.boolean_feature_indices]
        three_cat_features = features[:, self.three_cat_feature_indices]
        continuous_features = features[:, self.continuous_feature_indices]

        for c in range(self.num_classes):
            for ind_boolean, _ in enumerate(self.boolean_feature_indices): 
                boolean_feature = boolean_features[:, ind_boolean]
                class_boolean_feature = boolean_feature[labels == c]
                # since some cases failed to parse
                ny = (class_boolean_feature == 1).sum(dim=0).item() + (class_boolean_feature == -1).sum(dim=0).item()
                numerator = ((class_boolean_feature == 1).sum(dim=0).item() + self.boolean_priors[ind_boolean][c][1])
                denominator = (ny + sum(self.boolean_priors[ind_boolean][c]))
                if (abs(denominator)<1e-3):
                    denominator = 1e-3
                self.boolean_probs[c][ind_boolean] =  numerator/denominator 

            for ind_three_cat, _ in enumerate(self.three_cat_feature_indices):
                three_cat_feature = three_cat_features[:, ind_three_cat]
                class_three_cat_feature = three_cat_feature[labels == c]

                denominator = (class_three_cat_feature.shape[0] + sum(self.three_cat_priors[ind_three_cat][c]))
                if(abs(denominator)<1e-3):
                    denominator = 1e-3
                numerator_1 = ((class_three_cat_feature == -1).sum(dim=0).item() + self.three_cat_priors[ind_three_cat][c][0])
                self.three_cat_prob_1[c][ind_three_cat] = numerator_1/denominator
                numerator_2 = ((class_three_cat_feature == 0).sum(dim=0).item() + self.three_cat_priors[ind_three_cat][c][1])
                self.three_cat_prob_2[c][ind_three_cat] = numerator_2/denominator
                numerator_3 = ((class_three_cat_feature == 1).sum(dim=0).item() + self.three_cat_priors[ind_three_cat][c][2])
                self.three_cat_prob_3[c][ind_three_cat] = numerator_3/denominator

            for ind_continous, _ in enumerate(self.continuous_feature_indices):
                continuous_feature = continuous_features[:, ind_continous]
                class_continuous_feature = continuous_feature[labels == c]
                # self.mean[c][ind_continous] = (class_continuous_feature.sum(dim=0) + class_continuous_feature.shape[0]*self.continuous_priors[ind_continous][c][1]*self.continuous_priors[ind_continous][c][0]) / (class_continuous_feature.shape[0] + class_continuous_feature.shape[0]*self.continuous_priors[ind_continous][c][1])
                numerator = (class_continuous_feature.sum(dim=0) + self.continuous_priors[ind_continous][c][1]*self.continuous_priors[ind_continous][c][0])
                denominator = (class_continuous_feature.shape[0] + self.continuous_priors[ind_continous][c][1])
                if(abs(denominator)<1e-3):
                    denominator = 1e-3
                self.mean[c][ind_continous] = numerator / denominator
                self.std[c][ind_continous] = class_continuous_feature.std(dim=0)
                if(abs(self.std[c][ind_continous])<1e-3):
                    self.std[c][ind_continous] = 1e-3


    def predict(self, features):
        prediction_probs = []
        predictions = []
        for sample in features:
            class_probs = torch.zeros(2)
            for c in range(self.num_classes):
                log_class_prob = 0
                for i, ind in enumerate(self.boolean_feature_indices):
                    boolean_feature = sample[ind]
                    if (abs(boolean_feature-1.0)<1e-4):
                        log_class_prob += torch.log(self.boolean_probs[c][i])
                    elif (abs(boolean_feature-(-1.0))<1e-4):
                        log_class_prob += torch.log(1-self.boolean_probs[c][i])
                for i, ind in enumerate(self.three_cat_feature_indices):
                    three_cat_feature = sample[ind]
                    if (abs(three_cat_feature-1.0)<1e-4):
                        log_class_prob += torch.log(self.three_cat_prob_3[c][i])
                    elif (abs(three_cat_feature-0.0)<1e-4):
                        log_class_prob += torch.log(self.three_cat_prob_2[c][i])
                    elif (abs(three_cat_feature-(-1.0))<1e-4):
                        log_class_prob += torch.log(self.three_cat_prob_1[c][i])
                for i, ind in enumerate(self.continuous_feature_indices):
                    continuous_feature = sample[ind]
                    log_class_prob += norm.logpdf(continuous_feature, loc=self.mean[c][i], scale=self.std[c][i])
                class_probs[c] = log_class_prob
            
            class_probs = F.softmax(class_probs, dim=0)
            class_probs = [class_probs[0].item(), class_probs[1].item()]
            prediction_probs.append(class_probs)
            predicted_class = np.argmax(class_probs)
            predictions.append(int(predicted_class))

        return predictions, prediction_probs


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class NN2Layer(nn.Module):
    def __init__(self, input_size):
        super(NN2Layer, self).__init__()
        self.linear_1 = nn.Linear(input_size, 16)
        self.linear_2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #use relu
        out_1 = self.sigmoid(self.linear_1(x))
        out_2 = self.sigmoid(self.linear_2(out_1))
        return out_2


class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        input = input.squeeze()
        loss = (1.0 / self.gamma) * (target *  (1.0 - torch.pow(input, self.gamma)) + (1.0 - target) * (1.0 - torch.pow(1.0 - input, self.gamma)))
        return torch.mean(loss)
