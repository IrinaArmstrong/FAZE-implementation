
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from typing import List

import logging_handler
logger = logging_handler.get_logger(__name__)


class SizeEstimator(object):

    def __init__(self, model: nn.Module, input_size: List[int]):
        """
        Estimates the size of PyTorch models in memory
        for a given input size
        """
        self._model = model
        self._input_size = input_size

        # Calculate
        self._parameters_sizes = self.get_parameter_sizes()
        self._output_sizes = self.get_output_sizes()
        self._parameters_bits = self.calculate_parameters_bits()

    def get_parameter_sizes(self):
        """
        Get sizes of all parameters in `model`
        """
        sizes = []
        modules = list(self._model.modules())
        for i, module in enumerate(modules):
            # todo: make iterative call, to handle nested ModuleLists
            if isinstance(module, nn.ModuleList):
                for j, submodule in enumerate(module.modules()):
                    sizes.extend([np.array(param.size()) for param in submodule.parameters()])
            else:
                sizes.extend([np.array(param.size()) for param in module.parameters()])
        return sizes

    def get_output_sizes(self):
        """
        Run sample input through each layer to get output sizes
        """
        input_ = torch.Tensor(torch.FloatTensor(*self._input_size), volatile=True)
        modules = list(self._model.modules())
        out_sizes = []
        for i, module in enumerate(modules):
            out = module(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out
        return out_sizes

    def calculate_parameters_bits(self) -> float:
        """
        Calculate total number of bits to store `model` parameters
        """
        bits = 0
        total_bits = 0
        for param in self._parameters_sizes:
            # Choose dtype
            if param.dtype == torch.float16:
                bits = np.prod(param * 16)
            elif param.dtype == torch.bfloat16:
                bits = np.prod(param * 16)
            elif param.dtype == torch.float32:
                bits = np.prod(param * 32)
            elif param.dtype == torch.float64:
                bits = np.prod(param * 64)
            else:
                logger.error(f"Current version estimated only sizes of floating points parameters!")
            total_bits += bits
        return total_bits

    def calc_forward_backward_bits(self):
        """
        Calculate bits to store forward and backward pass
        """
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s))*self._bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits*2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self._input_size)) * self._bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total/8)/(1024**2)
        return total_megabytes, total