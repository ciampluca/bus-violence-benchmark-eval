import torch
import torch.fx
from torch import nn

class SlowFast(nn.Module):
    """
    Construct the SlowFast model from torch hub https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/
    
    """
    
    def __init__(
        self,
        model_pretrained=False,
        skip_weights_loading = False,
        cache_folder='./model_zoo',
        progress=True,
        out_channels=1,
        slowfast_alpha=4
    ):
        super().__init__()

        if skip_weights_loading:
            model_pretrained = False
        
        torch.hub.set_dir(cache_folder)
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=model_pretrained, verbose=progress)
            
        # replace final fully connected layer with a new one
        self.model.blocks._modules['6']._modules['proj'] = nn.Linear(2304, out_channels)

        self.slowfast_alpha = slowfast_alpha

    def forward(self, frames):
        fast_pathway = frames

        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            2,
            torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // self.slowfast_alpha
            ).long().to(frames.device),
        )

        frame_list = [slow_pathway, fast_pathway]
        pred = self.model(frame_list)

        # sigmoid
        cls_out = torch.sigmoid(pred)

        return cls_out
        
        
        
# Test code
def main():
    import torch
    
    input = torch.rand(4, 3, 32, 320, 320)
    model = SlowFast(model_pretrained=True)
    output = model(input)

    print("Input Shape: {}, Output Shape: {}".format(input.shape, output.shape))
    print("Output: {}".format(output))


if __name__ == "__main__":
    main()