# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_upsample_layer
from mmengine.model import BaseModule, xavier_init
from torch import Tensor
from mmcv.ops.carafe import CARAFEPack
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class Asymmetric_conv(nn.Module):
    """
    非对称卷积
    """

    def __init__(self, in_channel, out_channel, kernal_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, kernal_size), stride=1,
                               padding=(0, kernal_size // 2))
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(kernal_size, 1), stride=1,
                               padding=(kernal_size // 2, 0))

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        print(out2.shape)
        return out2


class Asymmetric_lateral(nn.Module):
    def __init__(self, in_channel, out_channel, kernal_size=(3, )):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernal_size = kernal_size

        # 第一个分支 1x1 3x3
        self.conv1x1_0 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)

        # 第二个分支 1x3
        # self.asy_conv_0 = Asymmetric_conv(self.in_channel, self.out_channel, self.kernal_size[0])
        self.asy_conv_0 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=(1, kernal_size[0]), stride=1,
                                    padding=(0, kernal_size[0] // 2))

        # 第三个分支 3x1
        # self.asy_conv_1 = Asymmetric_conv(self.in_channel, self.out_channel, self.kernal_size[1])
        self.asy_conv_1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=(kernal_size[0], 1), stride=1,
                                    padding=(kernal_size[0] // 2, 0))

        # 第四个分支 1x1
        self.conv1x1_1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride=1, padding=0)

        # 合并完经过3x3减少混叠效应
        self.conv_3x3_latest = nn.Conv2d(self.out_channel * 4, self.out_channel, kernel_size=3, stride=1, padding=1)

        self.attention = CBAMBlock(self.out_channel)

    def forward(self, x):
        gap1 = self.conv1x1_0(x)
        gap1 = self.conv3x3(gap1)

        gap2 = self.asy_conv_0(x)

        gap3 = self.asy_conv_1(x)

        gap4 = self.conv1x1_1(x)

        # out = gap1 + gap2 + gap3 + gap4
        out = torch.cat([gap1, gap2, gap3, gap4], dim=1)

        out = self.conv_3x3_latest(out)
        return self.attention(out)


@MODELS.register_module()
class MyFPN(BaseModule):
    r"""Feature Pyramid Network.


    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:

        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            num_outs: int,
            start_level: int = 0,
            end_level: int = -1,
            lateral_kernal_size: Tuple = (3, ),
            add_extra_convs: Union[bool, str] = False,
            relu_before_extra_convs: bool = False,
            no_norm_on_lateral: bool = False,
            conv_cfg: OptConfigType = None,
            norm_cfg: OptConfigType = None,
            act_cfg: OptConfigType = None,
            upsample_cfg: ConfigType = dict(mode='nearest'),
            init_cfg: MultiConfig = dict(
                type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        # --------------------  使用非对称卷积块替换 1x1 卷积   -------------------------------------- #
        self.lateral_kernal_size = lateral_kernal_size
        # --------------------------------------------------------------------------------------- #

        # -------------- ----------------------------------------------- #
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        # ------------------------------------------------------------------ #

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.conv_trans = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # l_conv = ConvModule(
            #     in_channels[i],
            #     out_channels,
            #     1,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
            #     act_cfg=act_cfg,
            #     inplace=False)

            l_conv = Asymmetric_lateral(in_channels[i], out_channels, self.lateral_kernal_size)

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i != self.backbone_end_level - 1:
                conv_t = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                self.conv_trans.append(conv_t)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                # 上一层特征经过 GavgPool 和 GmaxPool 后相加然后与下一层特征进行相乘，再与上采样后的特征现加
                x_max_pool = self.global_max_pool(laterals[i])
                x_avg_pool = self.global_avg_pool(laterals[i])
                x = torch.add(x_max_pool, x_avg_pool)
                weight = self.sigmoid(x)

                laterals[i - 1] = laterals[i - 1] * weight + self.conv_trans[i - 1](laterals[i])  # 上采样为反卷积

                # laterals[i - 1] = laterals[i - 1] * weight + F.interpolate(
                #     laterals[i], **self.upsample_cfg)
            else:
                x_max_pool = self.global_max_pool(laterals[i])
                x_avg_pool = self.global_avg_pool(laterals[i])
                x = torch.add(x_max_pool, x_avg_pool)
                weight = self.sigmoid(x)

                laterals[i - 1] = laterals[i - 1] * weight + self.conv_trans[i - 1](laterals[i])

                # prev_shape = laterals[i - 1].shape[2:]
                # laterals[i - 1] = laterals[i - 1] * weight + F.interpolate(
                #     laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # -------------------------- 输出热力图  ------------------------------------------- #
        # from mmengine.visualization import Visualizer
        # # visualizer = Visualizer()
        # image = mmcv.imread('E:/lrk/trail/datasets/SSDD/test/all/images/000739.png')
        # image = image.squeeze()
        # print(image.shape)     # (325, 503, 3)
        # # print(outs)
        # for i, out in enumerate(outs):
        #
        #
        #     print(i)
        #     visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
        #                             save_dir='./temp_dir/')
        #     drawn_img = visualizer.draw_featmap(out.squeeze(), image, channel_reduction='squeeze_mean',
        #                                         topk=1)
        #     visualizer.show(drawn_img)
        #
        #     # 会生成 temp_dir/vis_data/vis_image/feat_0.png
        #     visualizer.add_image('feat'+str(i), drawn_img)

        # -------------------------------------------------------------------------------- #
        # ----------------------------------- 输出特征图 ------------------------------------ #

        # 输出特征图
        # import matplotlib.pyplot as plt
        # import numpy as np
        # top = 12
        # for j, out in enumerate(outs):
        #     # [b,c,h,w] --> [c,h,w],从GPU转到CPU
        #     im = np.squeeze(out.cpu().detach().numpy())
        #
        #     c,h,w = im.shape
        #
        #     topk = min(c, top)
        #     sum_channel_featmap = torch.sum(Tensor(im), dim=(1, 2))
        #     _, indices = torch.topk(sum_channel_featmap, topk)
        #     im = im[indices]
        #
        #     # [c,h,w] --> [h,w,c]
        #     im = np.transpose(im, [1, 2, 0])
        #     plt.figure()
        #     # 前12个特征图
        #     for i in range(top):
        #         ax = plt.subplot(3, 4, i + 1)
        #         plt.imshow(im[:, :, i] * 255, cmap='gray')
        #         ax.tick_params(axis='both', which='both', length=0)  # 设置坐标轴数值不可见
        #         ax.axes.xaxis.set_visible(False)
        #         ax.axes.yaxis.set_visible(False)
        #     plt.axis('off')
        #     plt.savefig(f"E:/lrk/trail/logs/SAR/SSDD/feature_map/proposal_method/proposal_out_{j}.svg", format='svg', dpi=600, bbox_inches='tight')
        #     plt.show()

        # --------------------------------------------------------------------------------- #

        return tuple(outs)
