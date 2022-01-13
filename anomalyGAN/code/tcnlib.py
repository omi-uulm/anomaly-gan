import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
import copy

orig_deepcopy = getattr(nn.Conv1d, '__deepcopy__', None)

def __deepcopy__(self, memo):
    # save and delete all weightnorm weights on self
    weights = {}
    for hook in self._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            weights[hook.name] = getattr(self, hook.name)
            delattr(self, hook.name)
    # remove this deepcopy method, restoring the object's original one if necessary
    __deepcopy__ = self.__deepcopy__
    if orig_deepcopy:
        self.__deepcopy__ = orig_deepcopy
    else:
        del self.__deepcopy__
    # actually do the copy
    result = copy.deepcopy(self)
    # restore weights and method on self
    for name, value in weights.items():
        setattr(self, name, value)
    self.__deepcopy__ = __deepcopy__
    return result
# bind __deepcopy__ to the weightnorm'd layer


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        
        #orig_deepcopy = getattr(self.conv1, '__deepcopy__', None)
        self.conv1.__deepcopy__ = __deepcopy__.__get__(self.conv1, self.conv1.__class__)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        
        #orig_deepcopy = getattr(self.conv2, '__deepcopy__', None)
        self.conv2.__deepcopy__ = __deepcopy__.__get__(self.conv2, self.conv2.__class__)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

        
class TCNGenerator(nn.Module):
    def __init__(self, input_size=10, channels=20, num_layers=8, output_size=1,kernel_size=2, dropout=0.2, input_length=32, multi_variate=1, output_function=None):
        super().__init__()
        
        self.input_length = input_length
        
        self.output_function=output_function
        
        num_channels = [channels]*num_layers
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        self.linears = nn.ModuleList([nn.Linear(num_channels[-1], output_size) for _ in range(multi_variate)])
        
        self.init_weights()
        

    def init_weights(self):
        for linear in self.linears:
            linear.weight.data.normal_(0, 0.01)

    def forward(self, input):
        x, cond_input = input

        if cond_input is not None:
        
            x = torch.cat((x,cond_input),axis=2)
            

        
        x = x.permute(0,2,1)
        y1 = self.tcn(x)
             
        output = torch.stack([l(y1[:,:,-1]) for l in self.linears], axis=2)
        
        if self.output_function:
            output = self.output_function(output)
        
        return output
        

class TCNDiscriminator(nn.Module):
    def __init__(self, input_size=1, input_length=32,channels=20,num_layers=8, kernel_size=2, dropout=0.2, num_classes=1, softmax=False, wgan=False):
        super().__init__()
        
        self.input_length = input_length
        
        self.wgan = wgan
        
        num_channels = [channels]*num_layers
    
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        
        
        self.num_classes = num_classes
        
        self.linear1 = nn.Linear(num_channels[-1], num_classes)
        
        #self.linear2 = nn.Linear(input_length, 1)
        
        
        self.sigmoid = nn.Sigmoid()
        
        self.softmax = nn.Softmax(dim=1)
                
        self.init_weights()
        

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.01)

    def forward(self, input):
        x, cond_input = input    
               
        if cond_input is not None:
            x = torch.cat((x,cond_input),axis=2)
            
        x = x.transpose(1,2)
        y1 = self.tcn(x)
        
        if self.wgan:
            return self.linear1(y1[:,:,-1])
        
        if self.num_classes > 1: 
        
            return self.softmax(self.linear1(y1[:,:,-1]))
        
        return self.sigmoid(self.linear1(y1[:,:,-1]).squeeze(1))
