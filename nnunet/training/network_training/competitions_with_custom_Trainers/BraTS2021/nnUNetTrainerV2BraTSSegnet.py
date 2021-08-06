from typing import Optional
import torch
from torch import nn
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.segnet import SegNet, SmallSegNet
from nnunet.training.network_training.competitions_with_custom_Trainers.BraTS2020.nnUNetTrainerV2BraTSRegions import (
    nnUNetTrainerV2BraTSRegions,
)
from nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_focalLoss import (
    FocalLossMultiClass,
)
from nnunet.training.loss_functions.dice_loss import Tversky_and_CE_loss
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunet.training.dataloading.dataset_loading import unpack_dataset
import numpy as np
from nnunet.training.data_augmentation.data_augmentation_moreDA import (
    get_moreDA_augmentation,
)
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.network_architecture.neural_network import SegmentationNetwork


class nnUNetTrainerV2BraTSSegnet(nnUNetTrainerV2BraTSRegions):
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
        grid_num_conv_per_stage: Optional[int] = None,  # Grid: Number of convolutions
        grid_num_pool: Optional[int] = None,  # Grid: Number of poolings
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
        self.grid_num_conv_per_stage = grid_num_conv_per_stage
        self.grid_num_pool = grid_num_pool

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

        # NJ all args as for GenericUNet. But convolutional_pooling and convolutional_upsampling set to False!!
        self.network = SegNet(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes)
            if self.grid_num_pool is None
            else self.grid_num_pool,
            num_conv_per_stage=self.conv_per_stage
            if self.grid_num_conv_per_stage is None
            else self.grid_num_conv_per_stage,
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
            upscale_logits=False,
            convolutional_pooling=False,
            convolutional_upsampling=False,
        )
        if torch.cuda.is_available():
            self.network.cuda()

        # NJ Set inference_apply_nonlin as in nnUNetTrainerV2BraTSRegions
        self.network.inference_apply_nonlin = nn.Sigmoid()


class nnUNetTrainerV2BraTSSmallSegnet(nnUNetTrainerV2BraTSRegions):
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

        # NJ all args as for GenericUNet. But convolutional_pooling and convolutional_upsampling set to False!!
        self.network = SmallSegNet(
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
        )
        if torch.cuda.is_available():
            self.network.cuda()

        # NJ Set inference_apply_nonlin as in nnUNetTrainerV2BraTSRegions
        self.network.inference_apply_nonlin = nn.Sigmoid()


class nnUNetTrainerV2SegnetFocal(nnUNetTrainerV2BraTSSegnet):
    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
        )
        self.loss = FocalLossMultiClass()


class nnUNetTrainerV2SegNetTversky(nnUNetTrainerV2BraTSSegnet):
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
        super(nnUNetTrainerV2SegNetTversky, self).__init__(
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
        self.loss = Tversky_and_CE_loss(
            {"batch": self.batch_dice, "smooth": 1e-5, "do_bg": False}, {}
        )


#####
# Classes for grid search of segnet size.
# Parameters incluce number of pooling operations (5, 6) and number of convolutions (2, 3, 4).
#####


class nnUNetTrainerSegNetCustomSize(nnUNetTrainerV2BraTSSegnet):
    """
    Base class for grid search with pooling operations and number of convolutions.
    This class is necesarry to override the initialize function.

    Changes made compared to nnUNetTrainerV2BraTSRegions.initialize (See change 0.0.1):
        - Set 'net_pool_per_axis' to list of 3 * number of poolings. E.g. number poolings = 5 results in [5, 5, 5]
        - Set 'net_conv_kernel_sizes' to list of ([3] * dimenstionality) * number poolings + 1.
            E.g. number poolings = 3 in 3D results in [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
        - Set 'net_num_pool_op_kernel_sizes' like 'net_conv_kernel_sizes'
            except with kernel size = 2 instead of 3 and only number of poolings (instead number of poolings +1 )
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
        grid_num_conv_per_stage,
        grid_num_pool,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
            grid_num_conv_per_stage=grid_num_conv_per_stage,
            grid_num_pool=grid_num_pool,
        )

    def initialize(self, training=True, force_load_plans=False):
        """
        this is a copy of nnUNetTrainerV2's initialize. We only add the regions to the data augmentation
        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            # ################################## #
            #          Change 0.0.1.             #
            #  Change poolings and kernel sizes  #
            # ################################## #
            self.net_pool_per_axis = [self.grid_num_pool for _ in range(3)]
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (
                max(self.net_pool_per_axis) + 1
            )
            self.net_num_pool_op_kernel_sizes = [[2] * len(self.net_pool_per_axis)] * (
                max(self.net_pool_per_axis)
            )
            # ################################### #
            #           END CHANGE                #
            # ################################### #

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array(
                [True if i < net_numpool - 1 else False for i in range(net_numpool)]
            )
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans["data_identifier"] + "_stage%d" % self.stage,
            )
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!"
                    )

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr,
                    self.dl_val,
                    self.data_aug_params["patch_size_for_spatialtransform"],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    regions=self.regions,
                )
                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                    also_print_to_console=False,
                )
                self.print_to_log_file(
                    "VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                    also_print_to_console=False,
                )
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )
        self.was_initialized = True


class nnUNetTrainerSegNetPool5Conv3(nnUNetTrainerSegNetCustomSize):
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
        grid_num_conv_per_stage=3,
        grid_num_pool=5,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
            grid_num_conv_per_stage=grid_num_conv_per_stage,
            grid_num_pool=grid_num_pool,
        )


class nnUNetTrainerSegNetPool5Conv4(nnUNetTrainerSegNetCustomSize):
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
        grid_num_conv_per_stage=4,
        grid_num_pool=5,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
            grid_num_conv_per_stage=grid_num_conv_per_stage,
            grid_num_pool=grid_num_pool,
        )


class nnUNetTrainerSegNetPool6Conv2(nnUNetTrainerSegNetCustomSize):
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
        grid_num_conv_per_stage=2,
        grid_num_pool=6,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
            grid_num_conv_per_stage=grid_num_conv_per_stage,
            grid_num_pool=grid_num_pool,
        )


class nnUNetTrainerSegNetPool6Conv3(nnUNetTrainerSegNetCustomSize):
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
        grid_num_conv_per_stage=3,
        grid_num_pool=6,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
            grid_num_conv_per_stage=grid_num_conv_per_stage,
            grid_num_pool=grid_num_pool,
        )


class nnUNetTrainerSegNetPool6Conv4(nnUNetTrainerSegNetCustomSize):
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
        grid_num_conv_per_stage=4,
        grid_num_pool=6,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
            grid_num_conv_per_stage=grid_num_conv_per_stage,
            grid_num_pool=grid_num_pool,
        )
