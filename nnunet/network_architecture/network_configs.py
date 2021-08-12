from abc import ABC, abstractmethod
from dataclasses import dataclass, field, Field
from typing import List, Optional, Tuple
from torch import nn
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.generic_UNet import ConvDropoutNonlinNorm


def make_dict_field(value: dict) -> Field:
    return field(default_factory=lambda: value)


def compute_kernel_sizes(kernel_size, dim, poolings) -> List[Tuple]:
    return [tuple(kernel_size for _ in range(dim))] * (poolings + 1)


def compute_pad_sizes(conv_kernel_sizes) -> List[Tuple]:
    return [
        tuple([1 if element == 3 else 0 for element in kernel])
        for kernel in conv_kernel_sizes
    ]


@dataclass
class SegmentationNetworkConfigs(ABC):
    """Base Class for Segmentation methods"""

    input_channels: int
    base_num_features: int
    num_classes: int
    num_pool: int

    num_conv_per_stage: int = 2
    feat_map_mul_on_downscale: int = 2

    nonlin: nn.Module = nn.LeakyReLU

    final_nonlin = softmax_helper
    weightInitializer = InitWeights_He

    deep_supervision: bool = True
    dropout_in_localization: bool = False
    upscale_logits: bool = False
    convolutional_pooling: bool = False
    convolutional_upsampling: bool = False
    seg_output_use_bias: bool = False

    nonlin_kwargs: dict = make_dict_field({"negative_slope": 1e-2, "inplace": True})
    dropout_op_kwargs: dict = make_dict_field({"p": 0.5, "inplace": True})
    norm_op_kwargs: dict = make_dict_field(
        {"eps": 1e-5, "affine": True, "momentum": 0.1}
    )

    @property
    @abstractmethod
    def input_shape_must_be_divisible_by(self) -> int:
        pass

    @property
    @abstractmethod
    def pool_op_kernel_sizes(self) -> List[Tuple]:
        pass

    @property
    @abstractmethod
    def conv_kernel_sizes(self) -> List[Tuple]:
        pass

    @property
    @abstractmethod
    def conv_pad_sizes(self) -> List[Tuple]:
        pass


@dataclass
class SegmentationNetwork3dConfigs(SegmentationNetworkConfigs):
    """Class to store and process configs for 3D Segmentation Networks."""

    conv_op: nn.Module = nn.Conv3d
    norm_op: nn.Module = nn.InstanceNorm3d
    dropout_op: nn.Module = nn.Dropout3d
    pool_op = nn.MaxPool3d
    upsample_mode = "trilinear"
    max_num_features: Optional[int] = 320
    basic_block: nn.Module = ConvDropoutNonlinNorm

    @property
    def input_shape_must_be_divisible_by(self) -> int:
        return np.prod(self.pool_op_kernel_sizes, 0, dtype=np.int64)

    @property
    def pool_op_kernel_sizes(self) -> List[Tuple]:
        return compute_kernel_sizes(2, 3, self.num_pool)

    @property
    def conv_kernel_sizes(self) -> List[Tuple]:
        return compute_kernel_sizes(3, 3, self.num_pool)

    @property
    def conv_pad_sizes(self) -> List[Tuple]:
        return compute_pad_sizes(self.conv_kernel_sizes)


@dataclass
class SegmentationNetwork2dConfigs(SegmentationNetworkConfigs):
    """Class to store and process configs for 2D Segmentation Networks."""

    conv_op: nn.Module = nn.Conv2d
    norm_op: nn.Module = nn.InstanceNorm2d
    dropout_op: nn.Module = nn.Dropout2d
    pool_op = nn.MaxPool2d
    upsample_mode = "bilinear"
    max_num_features: Optional[int] = 480
    basic_block: nn.Module = ConvDropoutNonlinNorm

    @property
    def input_shape_must_be_divisible_by(self) -> int:
        return np.prod(self.pool_op_kernel_sizes, 0, dtype=np.int64)

    @property
    def pool_op_kernel_sizes(self) -> List[Tuple]:
        return compute_kernel_sizes(2, 2, self.num_pool)

    @property
    def conv_kernel_sizes(self) -> List[Tuple]:
        return compute_kernel_sizes(3, 2, self.num_pool)

    @property
    def conv_pad_sizes(self) -> List[Tuple]:
        return compute_pad_sizes(self.conv_kernel_sizes)
