import torch
from torch import nn
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.resnet import UResNet
from nnunet.training.network_training.competitions_with_custom_Trainers.BraTS2020.nnUNetTrainerV2BraTSRegions import (
    nnUNetTrainerV2BraTSRegions,
)

class nnUNetTrainerResNetEnc(nnUNetTrainerV2BraTSRegions):
    """
    SegNet trainer class. Implements training procedure using SegNet instead of generic UNet.

    Train model with:
    nnUNet_train 3d_fullres nnUNetTrainerV2BraTSSegnet Task500_Brats21 4 --npz
    """

    def __init__(
        self,
        plans_file,
        fold,
        output_folder,
        dataset_directory,
        batch_dice,
        stage,
        unpack_data,
        deterministic,
        fp16,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder=output_folder,
            dataset_directory=dataset_directory,
            batch_dice=batch_dice,
            stage=stage,
            unpack_data=unpack_data,
            deterministic=deterministic,
            fp16=fp16,
        )

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        dropout_op_kwargs = {"p": 0, "inplace": True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}

        self.network = UResNet(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            num_conv_per_stage=self.conv_per_stage,
            feat_map_mul_on_downscale=2,
            conv_op=conv_op,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=net_nonlin,
            nonlin_kwargs=net_nonlin_kwargs,
            deep_supervision=True,
            dropout_in_localization=False,
            final_nonlin=lambda x: x,
            weightInitializer=InitWeights_He(1e-2),
            pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
            conv_kernel_sizes=self.net_conv_kernel_sizes,
            upscale_logits=False,
            convolutional_pooling=False,
            convolutional_upsampling=False,
            residual_enc=True,
            residual_dec=False
        )
        if torch.cuda.is_available():
            self.network.cuda()

        # NJ Set inference_apply_nonlin as in nnUNetTrainerV2BraTSRegions
        self.network.inference_apply_nonlin = nn.Sigmoid()


class nnUNetTrainerResNetDec(nnUNetTrainerV2BraTSRegions):
    """
    SegNet trainer class. Implements training procedure using SegNet instead of generic UNet.

    Train model with:
    nnUNet_train 3d_fullres nnUNetTrainerV2BraTSSegnet Task500_Brats21 4 --npz
    """

    def __init__(
        self,
        plans_file,
        fold,
        output_folder,
        dataset_directory,
        batch_dice,
        stage,
        unpack_data,
        deterministic,
        fp16,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder=output_folder,
            dataset_directory=dataset_directory,
            batch_dice=batch_dice,
            stage=stage,
            unpack_data=unpack_data,
            deterministic=deterministic,
            fp16=fp16,
        )

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        dropout_op_kwargs = {"p": 0, "inplace": True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}

        self.network = UResNet(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            num_conv_per_stage=self.conv_per_stage,
            feat_map_mul_on_downscale=2,
            conv_op=conv_op,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=net_nonlin,
            nonlin_kwargs=net_nonlin_kwargs,
            deep_supervision=True,
            dropout_in_localization=False,
            final_nonlin=lambda x: x,
            weightInitializer=InitWeights_He(1e-2),
            pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
            conv_kernel_sizes=self.net_conv_kernel_sizes,
            upscale_logits=False,
            convolutional_pooling=False,
            convolutional_upsampling=False,
            residual_enc=True,
            residual_dec=True
        )
        if torch.cuda.is_available():
            self.network.cuda()

        # NJ Set inference_apply_nonlin as in nnUNetTrainerV2BraTSRegions
        self.network.inference_apply_nonlin = nn.Sigmoid()
