_base_ = [
    './fcos_r50_fpn_gn-head_50e_ssdd.py'
]

work_dir = "E:/lrk/trail/logs/SAR/SSDD/modify/fusion/fcos_r50_fusionfpn_v3_nearest_gn-head_50e_ssdd"

# -------------------------- 上采样使用转置卷积  ------------------------------------- #
# model = dict(
#     neck=dict(
#         type='FusionFPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         start_level=1,
#         add_extra_convs='on_output',  # use P5
#         num_outs=5,
#         relu_before_extra_convs=True),
# )
# -------------------------------------------------------------------------------------------- #

# ------------------- 上采样使用 bilinear 双线性插值  or  nearest 最近邻差值 -------------------------#
model = dict(
    neck=dict(
        type='FusionFPNV2',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True,
        upsample_cfg=dict(mode='nearest')
    ),
)
# -------------------------------------------------------------------------------------------- #

# -------------------------------------- 上采样使用 carafe --------------------------------------#
# model = dict(
#     neck=dict(
#         type='FusionFPNV1',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         start_level=1,
#         add_extra_convs='on_input',
#         num_outs=5,
#         relu_before_extra_convs=True,
#         upsample_cfg=dict(
#             type='carafe',
#             up_kernel=3,
#             up_group=1,
#             encoder_kernel=3,
#             encoder_dilation=1,
#             compressed_channels=64)
#     )
# )
# ---------------------------------------------------------------------------------------------- #
