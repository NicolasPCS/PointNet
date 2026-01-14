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

        print("Input dim", self.input_dim)
        print("Num features", self.num_features)
        # Shared MLP
        self.shared_mlp = nn.Sequential(
            #nn.BatchNorm1d(input_dim),
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU()
        )
        # Maxpooling layer
        # https://discuss.pytorch.org/t/is-torch-max-same-with-doing-maxpooling/45239
        self.mp = nn.MaxPool1d(1)
        # FC Layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, input): # torch.Size([2, 2500, 3])
        x = self.shared_mlp(input)
        
        x = self.mp(x)
        x = x.view(-1, 1024)
        print("x shape 1", x.size())

        # I use F.ReLU because I call it as a function
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        print("x shape 2", x.size())

        # Initialize the output matrix as an identity matrix (3x3?)
        id_matrix = Variable(torch.eye(3)).view(1,9).repeat(input.size()[0],1)
        print("id_matrix", id_matrix.size())

        if input.is_cuda:
            id_matrix = id_matrix.cuda()
        
        x += id_matrix
        print("_________", x.size())

        x = x.view(-1, 3, 3)
        print("_________", x.size())
        
        # Must return (1, 3, 3)
        return x


# Add regularization term
class TNetKd(torch.nn.Module):
    def __init__(self, k, num_features):
        super(TNetKd, self).__init__()

        self.k = k
        self.num_features = num_features

        self.shared_mlp = nn.Sequential(
            #nn.BatchNorm1d(input_dim),
            nn.Linear(64, 64),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU()
        )
        # Maxpooling layer
        self.mp = nn.MaxPool1d(self.num_features)
        # FC Layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(self.num_features)

    def forward(self, input):
        x = self.shared_mlp(input)
        x = self.mp(x)
        # I use F.ReLU because I call it as a function
        x = F.relu(self.bn1(self.fc1(x)))
        # Flatten for the final linear layer
        # -1 is used to automatically infer the number of columns
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        print("x shape from tnet3d", x.shape)

        # Initialize the output matrix as an identity matrix (kxk?)
        # TODO

        return 

num_features = 2500
class PointNetClassification(torch.nn.Module):
    def __init__(self, num_classes):
        super(PointNetClassification, self).__init__()

        self.TNet3d = TNet3d(3, num_features)
        
        self.first_smlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(num_features),
            nn.ReLU()
        )
        self.TNetKd = TNetKd(64, num_features)
        self.scnd_smlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
        )
        # Dropout layers are used for the last mlp in cls net
        self.lst_smlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Dropout(p=0.7)
        )
        # Maxpooling layer
        self.mp = nn.MaxPool1d(kernel_size=1)

    def forward(self, input):
        input = torch.transpose(input, 1, 2)
        # INPUT TRANSFORM
        # A natural solution is to align all input set to a canonical space
        # before feature extraction.
        # Affine transformation matrix
        x = self.TNet3d(input)
        # Do batch matrix-multitplication within input and TNet3d output
        print("PointNetClassification is", input.shape)
        print("PointNetClassification xs", x.shape)
        # input shape: (64, 3, 2500)
        # x shape: (3, 3)

        x = torch.bmm(input, x)
        #print(x.shape)
        # new x shape: (64, 3, 2500)?????

        x = self.first_smlp(x)

        # FEATURE TRANSFORM
        # We can insert another aligment network in point features and predict
        # a feature transformation matrix to align features from different point clouds
        x_kd = self.TNetKd(x)
        # Given that the transformation matrix in the feature space has much higher dimension
        # than the spatial transform matrix, they add a regularization term to the softmax 
        # training loss
        # TODO

        x = torch.bmm(x, x_kd)
        #print(x.shape)

        x = self.scnd_smlp(x)

        x = self.mp(x)

        x = self.lst_smlp(x)

        return torch.log_softmax(x, dim=1)