import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

# nn.Module: Base class for all neural network modules
class TNet3d(torch.nn.Module):
    def __init__(self, input_dim, num_features):
        super(TNet3d, self).__init__()
        self.input_dim = input_dim
        self.num_features = num_features

        #print("Input dim", self.input_dim)
        #print("Num features", self.num_features)
        
        # Shared MLP
        self.smlp1 = nn.Conv1d(self.input_dim, 64, 1)
        self.smlp2 = nn.Conv1d(64, 128, 1)
        self.smlp3 = nn.Conv1d(128, 1024, 1)

        # Maxpooling layer
        # https://discuss.pytorch.org/t/is-torch-max-same-with-doing-maxpooling/45239
        self.mp = nn.MaxPool1d(self.num_features)
        
        # FC Layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        # This FC layer is added because the expected output is a 3x3 matrix
        self.fc3 = nn.Linear(256, 9)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input): # torch.Size([2, 2500, 3])
        batchsize = input.size()[0]

        #print("input", input.size())

        # I use F.ReLU because I call it as a function
        x = F.relu(self.bn1(self.smlp1(input)))
        x = F.relu(self.bn2(self.smlp2(x)))
        x = F.relu(self.bn3(self.smlp3(x)))

        #print("x shape 1", x.size())

        x = self.mp(x)
        #print("x shape 2", x.size())
        x = x.view(-1, 1024)
        
        #print("x shape 3", x.size())

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        #print("x shape 4", x.size())

        # Initialize the output matrix as an identity matrix (3x3?)
        id_matrix = torch.eye(3).view(1,9).repeat(batchsize, 1)
        #print("id_matrix", id_matrix.size())

        if input.is_cuda:
            id_matrix = id_matrix.cuda()
        
        x += id_matrix
        #print("_________", x.size())

        x = x.view(-1, 3, 3)
        #print("_________", x.size())
        
        # Must return (B, 3, 3)
        return x


# Add regularization term
class TNetKd(torch.nn.Module):
    def __init__(self, k, num_features):
        super(TNetKd, self).__init__()

        self.k = k
        self.num_features = num_features
        
        # Shared MLP
        self.smlp1 = nn.Conv1d(64, 64, 1)
        self.smlp2 = nn.Conv1d(64, 128, 1)
        self.smlp3 = nn.Conv1d(128, 1024, 1)

        # Maxpooling layer
        self.mp = nn.MaxPool1d(self.num_features)

        # FC Layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.k*self.k)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        batchsize = input.size()[0]

        x = F.relu(self.bn1(self.smlp1(input)))
        x = F.relu(self.bn2(self.smlp2(x)))
        x = F.relu(self.bn3(self.smlp3(x)))

        x = self.mp(x)
        x = x.view(-1, 1024)

        #print("2. x shape 1", x.shape)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        #print("2. x shape 2", x.shape)

        # Initialize the output matrix as an identity matrix (kxk?)
        id_matrix = torch.eye(self.k).view(1,self.k**2).repeat(batchsize, 1)

        #print("2. x shape 3", id_matrix.shape)

        if x.is_cuda:
            id_matrix = id_matrix.cuda()
        
        x += id_matrix

        x = x.view(-1, self.k, self.k)

        return x

num_features = 2500
class PointNetClassification(torch.nn.Module):
    def __init__(self, num_classes):
        super(PointNetClassification, self).__init__()

        self.num_classes = num_classes

        # Input transform
        self.TNet3d = TNet3d(3, num_features)

        # First smlp
        self.smlp11 = nn.Conv1d(3, 64, 1)
        self.smlp12 = nn.Conv1d(64, 64,1)

        self.bn1 = nn.BatchNorm1d(64)

        # Feature transform
        self.TNetKd = TNetKd(64, num_features)

        # Second smlp
        self.smlp21 = nn.Conv1d(64, 64, 1)
        self.smlp22 = nn.Conv1d(64, 128, 1)
        self.smlp23 = nn.Conv1d(128, 1024, 1)

        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Last mlp
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Maxpooling layer
        self.mp = nn.MaxPool1d(num_features)

    def forward(self, input):
        # INPUT TRANSFORM
        # A natural solution is to align all input set to a canonical space
        # before feature extraction.
        # Affine transformation matrix
        x = self.TNet3d(input)
        
        # Do batch matrix-multitplication within input and TNet3d output
        #print("PointNetClassification is", input.shape)
        #print("PointNetClassification xs", x.shape)
        # input shape: (64, 3, 2500)
        # x shape: (64, 3, 3)

        x = torch.bmm(input.transpose(1, 2), x)
        #print("BMM", x.shape)
        # new x shape: (64, 2500, 3)

        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.smlp11(x)))
        x = F.relu(self.bn1(self.smlp12(x)))

        # FEATURE TRANSFORM
        # We can insert another aligment network in point features and predict
        # a feature transformation matrix to align features from different point clouds
        x_kd = self.TNetKd(x)
        # Given that the transformation matrix in the feature space has much higher dimension
        # than the spatial transform matrix, they add a regularization term to the softmax 
        # training loss
        # TODO

        x = x.transpose(1, 2)
        x = torch.bmm(x, x_kd)
        #print(x.shape)

        x = x.transpose(1, 2) 
        x = F.relu(self.bn1(self.smlp21(x)))
        x = F.relu(self.bn2(self.smlp22(x)))
        x = F.relu(self.bn3(self.smlp23(x)))

        x = self.mp(x)
        #print("________", x.size())
        x = x.view(-1, 1024)

        #print(x.size())

        # Dropout layers are used for the last mlp in cls net
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.dropout1d(x, 0.5)
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.dropout1d(x, 0.5)
        x = self.fc3(x)
        x = F.dropout1d(x, 0.7)

        #print("last x shape", x.size())
        
        return torch.log_softmax(x, dim=1)