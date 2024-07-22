from typing import Optional

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch

from brevitas.core.restrict_val import FloatRestrictValue
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
import brevitas.nn as qnn
from brevitas.nn.quant_layer import WeightQuantType
from brevitas.quant import Int8AccumulatorAwareWeightQuant
from brevitas.quant import Int8AccumulatorAwareZeroCenterWeightQuant
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat

class CommonIntWeightPerChannelQuant(Int8WeightPerTensorFloat):

    scaling_per_output_channel = True

class CommonIntAccumulatorAwareWeightQuant(Int8AccumulatorAwareWeightQuant):
    restrict_scaling_impl = FloatRestrictValue  # backwards compatibility
    bit_width = None


class CommonIntAccumulatorAwareZeroCenterWeightQuant(Int8AccumulatorAwareZeroCenterWeightQuant):
    bit_width = None

class CommonUintActQuant(Uint8ActPerTensorFloat):
    bit_width = None
    restrict_scaling_type = RestrictValueType.LOG_FP

def weight_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain('relu'))
        if layer.bias is not None:
            layer.bias.data.zero_()

class FloatLeNet(nn.Module):
    def __init__(self):
        super(FloatLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               stride=1,
                               padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels=6, 
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        
        self.fc1 = nn.Linear(400, 120,bias=True)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10,bias=True)
        
        
        self.apply(weight_init)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x,1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
        

class QuantLeNet(FloatLeNet):
    def __init__(self, weight_bit_width = 4, act_bit_width = 4, acc_bit_width = 32,weight_quant=CommonIntAccumulatorAwareWeightQuant):
        super(QuantLeNet, self).__init__()
        
        self.conv1 = qnn.QuantConv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               stride=1,
                               padding=0,
                               input_bit_width=act_bit_width,
                               input_quant=CommonUintActQuant,
                               weight_accumulator_bit_width=acc_bit_width,
                               weight_bit_width=weight_bit_width,
                               weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                               weight_quant=weight_quant)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = qnn.QuantReLU(inplace=True,act_quant=CommonUintActQuant,bit_width=act_bit_width)
        
        self.conv2 = qnn.QuantConv2d(in_channels=6, 
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=0,
                               input_bit_width=act_bit_width,
                               input_quant=CommonUintActQuant,
                               weight_accumulator_bit_width=acc_bit_width,
                               weight_bit_width=weight_bit_width,
                               weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                               weight_quant=weight_quant)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = qnn.QuantReLU(inplace=True,act_quant=CommonUintActQuant,bit_width=act_bit_width)
        
        
        self.fc1 = qnn.QuantLinear(400, 120,
                                bias=True,
                                input_bit_width=act_bit_width,
                                input_quant=CommonUintActQuant,
                                weight_accumulator_bit_width=acc_bit_width,
                                weight_bit_width=weight_bit_width,
                                weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                weight_quant=weight_quant)
        self.relu3 = qnn.QuantReLU(act_quant=CommonUintActQuant,bit_width=act_bit_width)
        self.fc2 = qnn.QuantLinear(120, 84,
                                bias=True,
                                input_bit_width=act_bit_width,
                                input_quant=CommonUintActQuant,
                                weight_accumulator_bit_width=acc_bit_width,
                                weight_bit_width=weight_bit_width,
                                weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                weight_quant=weight_quant)
        self.relu4 = qnn.QuantReLU(act_quant=CommonUintActQuant,bit_width=act_bit_width)
        self.fc3 = qnn.QuantLinear(84, 10,
                                bias=True,
                                input_bit_width=act_bit_width,
                                input_quant=CommonUintActQuant,
                                weight_accumulator_bit_width=acc_bit_width,
                                weight_bit_width=weight_bit_width,
                                weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                weight_quant=weight_quant)
        
        
        self.apply(weight_init)
