import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import models
from torchvision.models.alexnet import model_urls, AlexNet as AlexNetTorch
from torch.hub import load_state_dict_from_url


class ConvLSTM_AVSS2017(nn.Module):
    """
    Construct a model composed by an AlexNet backbone for feature extraction followed by ConvLSTM.
    Paper: S. Sudhakaran and O. Lanz - Learning to Detect Violent Videos using Convolutional Long Short-Term Memory - IEEE AVSS 2017
    Code inspired by: https://github.com/swathikirans/violence-recognition-pytorch
    """
    
    def __init__(
        self, 
        in_channels=3,
        out_channels=1,
        model_pretrained=False,
        skip_weights_loading = False,
        cache_folder='./model_zoo',
        progress=True,
        lstm_mem_size=256,
        kernel_size=3,
        device='cuda',
    ):
        
        super(ConvLSTM_AVSS2017, self).__init__()
        
        self.feat_extractor = AlexNetWrapper(
            model_pretrained=model_pretrained, 
            skip_weights_loading=skip_weights_loading, 
            cache_folder=cache_folder, 
            progress=progress,
        ).features_extractor

        self.temporal_module = ConvLSTMCell(
            input_size=256, 
            hidden_size=lstm_mem_size,
            device=device,
            kernel_size=kernel_size,
        )
        
        self.final_classifier = nn.Sequential(
            nn.Linear(3*3*lstm_mem_size, 1000), 
            nn.BatchNorm1d(1000), 
            nn.ReLU(), 
            nn.Linear(1000, 256), 
            nn.ReLU(),
            nn.Linear(256, 10), 
            nn.ReLU(), 
            nn.Linear(10, out_channels),
        )

    def forward(self, x):
        state = None
        x = torch.moveaxis(x, 1, 2)
        _, num_frames, _, _, _ = x.size()
        
        for i in range(num_frames):
            # Feed ConvLSTM with extracted features - one frame at a time
            x_ft = self.feat_extractor(x[:, i, :, :, :])
            state = self.temporal_module(x_ft, state)
            
        x = F.max_pool2d(state[0], kernel_size=2)
        x = self.final_classifier(x.view(x.size(0), -1))
        x = torch.sigmoid(x)
        
        return x
    

class ConvLSTMCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, kernel_size=3, stride=1, padding=1, device='cuda'):
        
        super(ConvLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.device = device
        
        torch.nn.init.xavier_normal(self.Gates.weight)
        torch.nn.init.constant(self.Gates.bias, 0)

    def forward(self, input_, prev_state):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (Variable(torch.zeros(state_size).to(self.device)), Variable(torch.zeros(state_size).to(self.device)))

        prev_hidden, prev_cell = prev_state
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        
        return hidden, cell
    
    
class AlexNetWrapper(AlexNetTorch):
    """
    Construct AlexNet model from torchvision implementation, removing final avgpool and fc layers
    """

    def __init__(
        self,
        model_pretrained=False,
        skip_weights_loading = False,
        cache_folder='./model_zoo',
        progress=True,
    ):
        
        super().__init__(
            num_classes=1000,   # for loading ImageNet pretraining
        )
        
        if skip_weights_loading:
            model_pretrained = False
            
        if model_pretrained:
            state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress, model_dir=cache_folder)
            self.load_state_dict(state_dict)
                        
        self.features_extractor = self._modules['features']



# Test code
def main():
    import torch
    
    device = 'cpu'
    
    input = torch.rand(2, 3, 16, 256, 256).to(device)
    model = ConvLSTM_AVSS2017(model_pretrained=True, device=device).to(device)
    print(model)
    output = model(input)

    print("Input Shape: {}, Output Shape: {}".format(input.shape, output.shape))
    print("Output: {}".format(output))


if __name__ == "__main__":
    main()