# Basic
import numpy as np
from typing import List
from collections import namedtuple

# Torch utils
import torch
import torch.nn as nn

Parameter = namedtuple('Parameter', ['size', 'bits'],
                       defaults=[np.asarray((0, 0)), 32])


class SizeEstimator(object):

    def __init__(self, model: nn.Module, input_size: List[int],
                 input_n_bits: int = 32):
        """
        Estimates the size of PyTorch models in memory
        for a given input size and data precision, measured in bits.
        So default input type of torch.float32 equals to 32 bits precision.
        """
        self._model = model
        self._input_size = input_size
        self._input_n_bits = input_n_bits

        # Calculate
        self._parameters_sizes = self._get_parameter_sizes()
        self._output_sizes = self._get_output_sizes()
        self._parameters_bits = self._calculate_parameters_weight()
        self._forward_backward_bits = self._calculate_forward_backward_weight()
        self._input_weight = self._calculate_input_weight()

    def _get_parameter_sizes(self) -> List[Parameter]:
        """
        Get sizes of all parameters in `model`
        """
        sizes = []
        modules = list(self._model.modules())[1:]
        for i, module in enumerate(modules):
            if isinstance(module, nn.ModuleList):
                # To not to estimate inner sub-modules twice!
                continue
            else:
                sizes.extend([Parameter(size=np.asarray(param.size()),
                                        bits=self.__get_parameter_bits(param))
                              for param in module.parameters()])
        return sizes

    def _get_output_sizes(self) -> List[Parameter]:
        """
        Run sample input through each layer to get output sizes
        """
        input_ = torch.Tensor(torch.FloatTensor(*self._input_size), volatile=True)
        modules = list(self._model.modules())[1:]
        out_sizes = []
        for i, module in enumerate(modules):
            out = module(input_)
            out_sizes.append(Parameter(size=np.asarray(out.size()),
                                       bits=self.__get_parameter_bits(out)))
            input_ = out
        return out_sizes

    def _calculate_parameters_weight(self) -> float:
        """
        Calculate total number of bits to store `model` parameters
        """
        total_bits = 0
        for param in self._parameters_sizes:
            total_bits += np.prod(param.size) * param.bits
        return total_bits

    @staticmethod
    def __get_parameter_bits(param: torch.Tensor) -> int:
        """
        Calculate total number of bits to store `model` parameters
        """
        # Choose dtype
        if param.dtype == torch.float16:
            return 16
        elif param.dtype == torch.bfloat16:
            return 16
        elif param.dtype == torch.float32:
            return 32
        elif param.dtype == torch.float64:
            return 64
        else:
            print(f"Current version estimated only sizes of floating points parameters!")
            return 32

    def _calculate_forward_backward_weight(self) -> float:
        """
        Calculate bits to store forward and backward pass
        """
        total_bits = 0
        for out in self._output_sizes:
            # forward pass
            f_bits = np.prod(out.size) * out.bits
            total_bits += f_bits

        # Multiply by 2 for both forward and backward
        return total_bits * 2

    def _calculate_input_weight(self) -> float:
        """
        Calculate bits to store single input sequence.
        """
        return np.prod(np.array(self._input_size)) * self._input_n_bits

    def estimate_total_size(self) -> float:
        """
        Estimate model size in memory in megabytes and bits.
        """
        total = self._input_weight + self._parameters_bits + self._forward_backward_bits
        total_bytes = (total / 8)
        total_megabytes = total_bytes / (1024**2)
        print(f"Model size is: {total} bits, {total_bytes} bytes, {total_megabytes} Mb.")
        return total_megabytes