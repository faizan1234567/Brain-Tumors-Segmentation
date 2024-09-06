import torch
import torch.nn as nn
from research.models.TransUNet3D.networks.transunet3d_model import Generic_TransUNet_max_ppbp
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input = torch.rand(1, 4, 64, 64, 64).to(device)
model = Generic_TransUNet_max_ppbp(4, 30, 3, num_pool=999, conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, dropout_op=nn.Dropout3d, 
                                   deep_supervision=False).to(device)

output = model(input)
print(output.shape)