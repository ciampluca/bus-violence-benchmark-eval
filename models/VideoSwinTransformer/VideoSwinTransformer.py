from collections import OrderedDict
from .backbone import SwinTransformer3D
import torch
from torch import nn
from torch.hub import load_state_dict_from_url

MODEL_URLS = {
    # swin base model
    "swin_base_ssv2": "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth",
    "swin_base_kinetics400": "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth"
}

class VideoSwinTransformer(nn.Module):
    """
    Construct a swin transformer.
    Ref: Video Swin Transformer - https://arxiv.org/abs/2106.13230
    """
    
    def __init__(self,
        model_pretrained=True,
        backbone='swin_base',
        progress=True,
        cache_folder='./model_zoo',
        skip_weights_loading=False,
        **kwargs
    ):
        super().__init__()

        if skip_weights_loading:
            model_pretrained = False

        # Backbone implementation from https://github.com/haofanwang/video-swin-transformer-pytorch
        self.swin_transformer_3d = SwinTransformer3D(**kwargs)
        if model_pretrained:
            checkpoint = load_state_dict_from_url(MODEL_URLS[backbone], progress=progress, model_dir=cache_folder)
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'backbone' in k:
                    name = k[9:]
                    new_state_dict[name] = v 

            self.swin_transformer_3d.load_state_dict(new_state_dict)

        # Classification layer
        self.cls = nn.Linear(1024, 1)


    def forward(self, x):
        x = self.swin_transformer_3d(x)

        # mean pooling
        x = x.mean(dim=[2,3,4]) # [batch_size, hidden_dim]

        # project
        cls_out = self.cls(x)

        # sigmoid
        cls_out = torch.sigmoid(cls_out)

        return cls_out

        
# Test code
def main():
    import torch
    
    input = torch.rand(1, 3, 32, 224, 224).to('cuda')
    model = VideoSwinTransformer(
        model_pretrained=True,
        embed_dim=128, 
        depths=[2, 2, 18, 2], 
        num_heads=[4, 8, 16, 32], 
        patch_size=(2,4,4), 
        window_size=(16,7,7), 
        drop_path_rate=0.4, 
        patch_norm=True)
    model.to('cuda')

    output = model(input)

    print("Input Shape: {}, Output Shape: {}".format(input.shape, output.shape))
    # print("Output: {}".format(output))


if __name__ == "__main__":
    main()

        
