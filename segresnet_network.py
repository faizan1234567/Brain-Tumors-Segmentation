# architecture test and how the reprsentation are calculated
import torch
from research.models.SegResNet.segesnet import SegResNet

# Define MRI crop
input = torch.rand(1, 4, 160, 192, 128)

# Define model
spatial_size = 3
in_channels = 4
num_classes = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model =         SegResNet(spatial_dims=spatial_size, 
                          init_filters=32, 
                          in_channels=in_channels, 
                          out_channels=num_classes, 
                          dropout_prob=0.2, 
                          blocks_down=(1, 2, 2, 4), 
                          blocks_up=(1, 1, 1)).to(device)

output = model(input)

print(output.shape)
