from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# Class for input transformation - Predicts a transformation matrix
# Three 1D convolutions
class TNET3d(nn.Module):
    def __init__(self):
        super(TNET3d, self).__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # Fully-connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        # Activation function
        self.relu = nn.ReLU()

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        batchsize = x.size()[0]
        # Per-point features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Max pooling to agregate global features (symmetric operation)
        # To mantain permutation-invariance
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # The output matrix is initialized as an dentity matrix. Allow the
        # network to learn small geometric corrections over a canonical pose
        identity = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            identity = identity.cuda()
        x += identity
        x = x.view(-1,3,3)
        return x


# Class for feature transformation - Predicts a transformation matrix
# Three 1D convolutions with k inputs
class TNETkd(nn.Module):
    def __init__(self, k=64):
        super(TNETkd, self).__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # Fully-connected layers
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k*k)
        # Activation function
        self.relu = nn.ReLU()

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        # Per-point features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Max pooling to agregate global features (symmetric operation)
        # To mantain permutation-invariance
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # The output matrix is initialized as an dentity matrix. Allow the
        # network to learn small geometric corrections over a canonical pose
        identity = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            identity = identity.cuda()
        x += identity
        x = x.view(-1, self.k, self.k)
        return x

# Class to extract features, both input and features transformations
class PointNetFeatures(nn.Module):
    def __init__(self, global_features = True, feature_transform = False):
        super(PointNetFeatures, self).__init__()
        self.stn = TNET3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_features = global_features
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = TNETkd(k=64)
    
    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        # bmm: Batch Matrix-Matrix Product
        # Compute matrix multiplication by batches
        # (b, n, m) x (b, m, p) = (b, n, p)
        print("X", x.shape)
        print("TRANS", trans.shape)
        print(trans)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else: 
            trans_feat = None
        
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_features:
            return x, trans, trans_feat
        else: 
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetClassification(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetClassification, self).__init__()
        self.feature_transform = feature_transform
        self.features = PointNetFeatures(global_features=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x, trans, trans_feat = self.features(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

# DENSE: Segmentation
class PointNetDenseClassification(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseClassification, self).__init__()
        return

# Force the learned matrix to be orthogonal
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) -I, dim=(1, 2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 2, 2500))
    trans = TNET3d()
    out = trans(sim_data)
    print('TNET3d', out.size())
    print('LOSS', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNETkd(k=64)
    out = trans(sim_data_64d)
    print('TNETkd', out.size())
    print('LOSS', feature_transform_regularizer(out))

    point_features = PointNetFeatures(global_features=True)
    out, _, _ = point_features(sim_data)
    print('GLOBAL FEATURES', out.size())

    point_features = PointNetFeatures(global_features=False)
    out, _, _ = point_features(sim_data)
    print("POINT FEATURES", out.size())

    classification = PointNetClassification(k = 5)
    out, _, _ = classification(sim_data)
    print("CLASSIFICATION", out.size())