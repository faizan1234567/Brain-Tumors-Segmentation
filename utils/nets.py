
from monai.networks.nets import SwinUNETR, SegResNet, VNet, BasicUNetPlusPlus, AttentionUnet, DynUNet, UNETR
from research.models.ResUNetpp.model import ResUnetPlusPlus


class NeuralNet:
    """pick the model for training"""
    def __init__(self, model_name: str, device = None, dims=3, in_channels=4, out_channels=3):
        self.model_name = model_name
        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self._all_models = {
            "SegResNet": SegResNet(spatial_dims=3, 
                                   init_filters=32, 
                                   in_channels=4, 
                                   out_channels=3, 
                                   dropout_prob=0.2, 
                                   blocks_down=(1, 2, 2, 4), 
                                   blocks_up=(1, 1, 1)).to(self.device),

            "VNet":VNet(spatial_dims=3, 
                        in_channels=4, 
                        out_channels=3,
                        dropout_dim=1,
                        bias= False
                        ).to(self.device),

            "AttentionUNet": AttentionUnet(spatial_dims=3, 
                                           in_channels=4, 
                                           out_channels=3, 
                                           channels= (8, 16, 32, 64, 128), 
                                           strides = (2, 2, 2, 2),
                                           ).to(self.device),

            "ResUnetPlusPlus": ResUnetPlusPlus(in_channels=4,
                                         out_channels=3).to(self.device),

            "UNETR": UNETR(in_channels=self.in_channels, 
                           out_channels=self.out_channels, 
                           img_size=(128,128,128), 
                           proj_type='conv', 
                           norm_name='instance').to(self.device),

            "SwinUNETR": SwinUNETR(
                    img_size=128,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    feature_size=48,
                    drop_rate=0.1,
                    attn_drop_rate=0.2,
                    dropout_path_rate=0.1,
                    spatial_dims=3,
                    use_checkpoint=False,
                    use_v2=False).to(device), 
            
        
        if self.model_name == "DynUNet":
            dyn_model = self.build_dynunet()
            self._all_models["DynUNet"] = dyn_model.to(self.device)
        
    def select_model(self):
        return self._all_models[self.model_name]
    
    def get_dynunet_params(self, depth=6, min_fmap=2, patch_size=[128, 128, 128], spacings=[1.0, 1.0, 1.0]):
        strides, kernels, sizes = [], [], patch_size[:]
        while True:
            spacing_ratio = [spacing / min(spacings) for spacing in spacings]
            stride = [
                2 if ratio <= 2 and size >= 2 * min_fmap else 1 for (ratio, size) in zip(spacing_ratio, sizes)
            ]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
            if len(strides) == depth:
                break
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        return kernels, strides, patch_size
    
    def build_dynunet(self, filters = [64, 96, 128, 192, 256, 384, 512]):
        kernels, strides, patch_size = self.get_dynunet_params()
      
       
        self.model = DynUNet(
            self.dims,
            self.in_channels,
            self.out_channels,
            kernels,
            strides,
            strides[1:],
            filters=filters,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
            deep_supervision=True,
            deep_supr_num=2,
            res_block=False,
            trans_bias=True,
        )
        return self.model