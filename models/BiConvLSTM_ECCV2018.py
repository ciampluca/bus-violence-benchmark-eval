import torch
from torch import nn
from torchvision.models.vgg import cfgs, model_urls, make_layers, VGG as VGGTorch
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F


class BiConvLSTM_ECCV2018(nn.Module):
    """
    Construct a model composed by a VGG13 backbone for feature extraction followed by Bidirectional ConvLSTM.
    Paper: A. Hanson et al. - Bidirectional Convolutional LSTM for the Detection of Violence in Videos - IEEE ECCV 2018
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        model_pretrained=False,
        skip_weights_loading = False,
        cache_folder='./model_zoo',
        progress=True,
        batch_norm=True,
        lstm_mem_size=256,
        kernel_size=3,
    ):
        
        super(BiConvLSTM_ECCV2018, self).__init__()
        
        self.feat_extractor = VGGWrapper(
            model_pretrained=model_pretrained, 
            skip_weights_loading=skip_weights_loading, 
            cache_folder=cache_folder, 
            progress=progress,
            batch_norm=batch_norm,
        ).features_extractor
        
        # TODO not sure if in the paper lstm_mem_size is 256 or 512
        self.temporal_module = ConvLSTM(
            input_dim=512, 
            hidden_dim=lstm_mem_size, 
            kernel_size=(kernel_size, kernel_size), 
            batch_first=True,
        )
        
        self.final_classifier = nn.Sequential(
            nn.Linear(7*7*lstm_mem_size*2, 1000), 
            nn.BatchNorm1d(1000), 
            nn.Tanh(), 
            nn.Linear(1000, 256), 
            nn.Tanh(),
            nn.Linear(256, 10), 
            nn.Tanh(), 
            nn.Linear(10, out_channels),
        )
    
    def forward(self, x):
        x = torch.moveaxis(x, 1, 2)
        _, num_frames, _, _, _ = x.size()
        
        output = []
        for i in range(num_frames):
            # Feature Extraction - one frame at a time
            x_ft = self.feat_extractor(x[:, i, :, :, :])
            output.append(x_ft)
        
        x = torch.stack(output, dim=1)
        x = self.temporal_module(x) 
        
        x, _ = torch.max(x, dim=1) 
            
        x = F.avg_pool2d(x, kernel_size=2)
        x = self.final_classifier(x.view(x.size(0), -1))
        x = torch.sigmoid(x)
        
        return x
    
    
class ConvLSTMCell(nn.Module):
    """
    Implementation from https://github.com/ndrplz/ConvLSTM_pytorch
    """

    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        kernel_size, 
        bias
    ):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
        
        
class ConvLSTM(nn.Module):
    """
    Implementation adapted from from https://github.com/ndrplz/ConvLSTM_pytorch and https://github.com/KimUyen/ConvLSTM-Pytorch
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:

    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        kernel_size, 
        batch_first=False, 
        bias=True, 
    ):
        
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.batch_first = batch_first
        self.bias = bias

        self.cell_fw = ConvLSTMCell(input_dim=self.input_dim,
                                        hidden_dim=self.hidden_dim,
                                        kernel_size=self.kernel_size,
                                        bias=self.bias)
        
        self.cell_bw = ConvLSTMCell(input_dim=self.input_dim,
                                        hidden_dim=self.hidden_dim,
                                        kernel_size=self.kernel_size,
                                        bias=self.bias)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state, hidden_state_inv = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
        
        # LSTM forward direction
        input_fw = input_tensor
        h, c = hidden_state
        output_inner = []
        for t in range(seq_len):
            h, c = self.cell_fw(input_tensor=input_fw[:, t, :, :, :], cur_state=[h, c])        
            output_inner.append(h)

        output_inner = torch.stack(output_inner, dim=1)
        last_state = [h, c]
            
        # LSTM inverse direction
        input_inv = input_tensor
        h_inv, c_inv = hidden_state_inv
        output_inv = []
        for t in range(seq_len-1, -1, -1):
            h_inv, c_inv = self.cell_bw(input_tensor=input_inv[:, t, :, :, :], cur_state=[h_inv, c_inv])
            output_inv.append(h_inv)
            
        output_inv.reverse()
        output_inv = torch.stack(output_inv, dim=1)
        layer_output = torch.cat((output_inner, output_inv), dim=2)
        last_state_inv = [h_inv, c_inv]

        return layer_output

    def _init_hidden(self, batch_size, image_size):
        init_states_fw = self.cell_fw.init_hidden(batch_size, image_size)
        init_states_bw = self.cell_bw.init_hidden(batch_size, image_size)
            
        return init_states_fw, init_states_bw

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')


class VGGWrapper(VGGTorch):
    """
    Construct VGG13 model from torchvision implementation, removing final maxpool, avgpool and fc layers
    Ref: 
    """

    def __init__(
        self,
        model_pretrained=False,
        skip_weights_loading = False,
        cache_folder='./model_zoo',
        progress=True,
        batch_norm=False,
    ):
        
        super().__init__(
            features=make_layers(cfgs['B'], batch_norm=batch_norm),
            num_classes=1000,   # for loading ImageNet pretraining
        )
        
        if skip_weights_loading:
            model_pretrained = False
            
        if model_pretrained:
            model_name = "vgg13" + "_bn" if batch_norm else "vgg13"
            state_dict = load_state_dict_from_url(model_urls[model_name], progress=progress, model_dir=cache_folder)
            self.load_state_dict(state_dict)
            
        self.features_extractor = self._modules['features'][:-1]



# Test code
def main():
    import torch
    
    device = 'cpu'
    
    input = torch.rand(2, 3, 16, 224, 224).to(device)
    model = BiConvLSTM_ECCV2018(model_pretrained=True).to(device)
    print(model)
    output = model(input)

    print("Input Shape: {}, Output Shape: {}".format(input.shape, output.shape))
    print("Output: {}".format(output))


if __name__ == "__main__":
    main()
        