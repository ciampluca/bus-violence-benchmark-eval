import torchvision
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.video.resnet import VideoResNet as VideoResNetTorch, BasicBlock, R2Plus1dStem, Conv2Plus1D


MODEL_URLS = {
    # 18 layer deep ResNet(2+1)D network pretrained on Kinetics
    "resnet18": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
}


class ResNet2Plus1D18Wrapper(VideoResNetTorch):
    """
    Construct 18 layer Resnet(2+1)D model using the torchvision implementation.
    Ref: Do Tran et al. - A Closer Look at Spatiotemporal Convolutions for Action Recognition - https://arxiv.org/abs/1711.11248
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        backbone='resnet18',
        model_pretrained=False,
        skip_weights_loading = False,
        cache_folder='./model_zoo',
        progress=True,
    ):
        
        if skip_weights_loading:
            model_pretrained = False
        
        block = BasicBlock
        conv_makers = [Conv2Plus1D] * 4
        layers = [2, 2, 2, 2]
        stem = R2Plus1dStem
        
        super().__init__(
            block=block,
            layers=layers,
            conv_makers=conv_makers,
            stem=stem,
            num_classes=400,    # for loading Kinetics pretraining
        )
        
        if model_pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS[backbone], progress=progress, model_dir=cache_folder)
            self.load_state_dict(state_dict)
            
        # replace final fully connected layer with a new one
        fc = nn.Linear(512 * block.expansion, out_channels)
        self.fc = fc
        
        
        
# Test code
def main():
    import torch
    
    input = torch.rand(4, 3, 96, 112, 112)
    model = ResNet2Plus1D18Wrapper(model_pretrained=True)
    output = model(input)

    print("Input Shape: {}, Output Shape: {}".format(input.shape, output.shape))
    print("Output: {}".format(output))


if __name__ == "__main__":
    main()



