_base_ = [
    './fcos_r50_fpn_gn-head_50e_ssdd.py'
]

work_dir = "E:/lrk/trail/logs/SAR/SSDD/modify/lateral/fcos_r50_myfpn_k7_gn-head_50e_ssdd_v3"

# -------------------------- 横向使用非对称卷积  ------------------------------------- #
model = dict(
    neck=dict(
        type='MyFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        lateral_kernal_size=(5, ),
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
)
# -------------------------------------------------------------------------------------------- #
