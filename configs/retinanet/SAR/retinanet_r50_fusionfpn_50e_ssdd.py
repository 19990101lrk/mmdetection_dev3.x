_base_ = [
    './retinanet_r50_fpn_50e_ssdd.py'
]

work_dir = "E:/lrk/trail/logs/SAR/SSDD/modify/retinanet_r50_fusionfpn_50e_ssdd"

model = dict(
    neck=dict(
        type='FusionFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        upsample_cfg=dict(
            type='carafe',
            up_kernel=3,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64)
    )

)
