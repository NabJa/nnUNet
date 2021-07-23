from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import nn
import numpy as np
from nnunet.network_architecture.initialization import InitMethod, InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.generic_UNet import ConvDropoutNonlinNorm


@dataclass
class SegmentationNetworkConfigs:
    """Class to store and process configs for Segmentation Networks."""

    input_channels: int
    base_num_features: int
    num_classes: int
    num_pool: int
    num_conv_per_stage: int = 2
    feat_map_mul_on_downscale: int = 2
    conv_op: nn.Module = nn.Conv3d
    norm_op: nn.Module = nn.InstanceNorm3d
    norm_op_kwargs: Optional[dict] = None
    dropout_op: nn.Module = nn.Dropout3d
    dropout_op_kwargs: Optional[dict] = None
    nonlin: nn.Module = nn.LeakyReLU
    nonlin_kwargs: Optional[dict] = None
    deep_supervision: bool = True
    dropout_in_localization: bool = False
    final_nonlin = softmax_helper
    weightInitializer: InitMethod = InitWeights_He
    pool_op_kernel_sizes: Optional[Tuple] = None
    conv_kernel_sizes: Optional[Tuple] = None
    upscale_logits: bool = False
    convolutional_pooling: bool = False
    convolutional_upsampling: bool = False
    max_num_features: Optional[int] = None
    basic_block: nn.Module = ConvDropoutNonlinNorm
    seg_output_use_bias: bool = False

    def __post_init__(self) -> None:
        self.parse_kwargs()
        self.parse_conv_op()
        self.parse_max_num_features()
        self.conv_pad_sizes = self.compute_pad_sizes()
        self.input_shape_must_be_divisible_by = self.comput_input_shape_divisor()

    def parse_kwargs(self) -> None:
        if self.nonlin_kwargs is None:
            self.nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        if self.dropout_op_kwargs is None:
            self.dropout_op_kwargs = {"p": 0.5, "inplace": True}
        if self.norm_op_kwargs is None:
            self.norm_op_kwargs = {"eps": 1e-5, "affine": True, "momentum": 0.1}

    def parse_conv_op(self) -> None:
        if self.conv_op == nn.Conv2d:
            self.set_conv_pool_op(2)
            self.set_conv_pool_kernels(2)
        elif self.conv_op == nn.Conv3d:
            self.set_conv_pool_op(3)
            self.set_conv_pool_kernels(3)
        else:
            raise ValueError(
                "unknown convolution dimensionality, conv op: %s" % str(self.conv_op)
            )

    def set_conv_pool_op(self, dimensionality=3):
        if dimensionality == 3:
            self.upsample_mode = "trilinear"
            self.pool_op = nn.MaxPool3d
        elif dimensionality == 2:
            self.upsample_mode = "bilinear"
            self.pool_op = nn.MaxPool2d
        else:
            raise ValueError("Unknown dimensionality", dimensionality)

    def set_conv_pool_kernels(self, dimensionality=3) -> None:
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = self.compute_kernel_sizes(3, dimensionality)
        if self.pool_op_kernel_sizes is None:
            self.pool_op_kernel_sizes = self.compute_kernel_sizes(2, dimensionality)

    def set_conv_op(self, dimensionality=3) -> None:
        conv_kernel = tuple([3] * dimensionality)
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [conv_kernel] * (self.num_pool + 1)

    def compute_kernel_sizes(self, size=2, dimensionality=3) -> List[Tuple]:
        kernel = tuple([size] * dimensionality)
        return [kernel] * (self.num_pool + 1)

    def compute_pad_sizes(self) -> List[Tuple]:
        def _kernel_to_padding(kernel):
            return tuple([1 if kernel_size == 3 else 0 for kernel_size in kernel])

        return [_kernel_to_padding(krnl) for krnl in self.conv_kernel_sizes]

    def comput_input_shape_divisor(self) -> int:
        return np.prod(self.pool_op_kernel_sizes, 0, dtype=np.int64)

    def parse_max_num_features(self):
        if self.max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = 320
            else:
                self.max_num_features = 480
