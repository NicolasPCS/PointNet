# Import the library
import torch
import torch.nn as nn

# Define the input data as a 3-D tensor
x = torch.tensor([[[0.,1.,2.],
                   [3.,4.,5.],
                   [6.,7.,8.],
                   [6.,7.,8.]],
                   [[0.,1.,2.],
                   [3.,4.,5.],
                   [6.,7.,8.],
                   [6.,7.,8.]]])

print('input_data:', x.shape)
print(x.size()[-1])
# Create an instance of the nn.Linear class
output_size = 5
""" linear_layer = nn.Linear(in_features=x.size()[-1],
                         out_features=output_size,
                         bias = True,
                         dtype= torch.float) """

linear_layer = nn.Sequential(
            #nn.BatchNorm1d(input_dim),
            nn.Linear(x.size()[-1], 64, 64),
            nn.BatchNorm1d(x.size()[-2]),
            nn.ReLU(),
            nn.Linear(64, 128, 128),
            nn.BatchNorm1d(x.size()[-2]),
            nn.ReLU(),
            nn.Linear(128, 1024, 1024),
            nn.BatchNorm1d(x.size()[-2]),
            nn.ReLU()
        )

# Apply the linear transformation to the input data
y = linear_layer(x)
#print the outputs
print('\noutput:\n',y)

#print the outputs shape
print('\noutput Shape:', y.shape)