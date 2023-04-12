_base_ = [
    '../fcos_r50_fpn_gn-head_50e_ssdd.py'
]

work_dir = "E:/lrk/trail/logs/SAR/SSDD/modify/fpn/fcos_r50_bifpn_gn-head_50e_ssdd"

# ------------------- FCOS + BiFPN -------------------------#

model = dict(
    neck=dict(
        _delete_=True,
        type='BiFPN',
        num_stages=6,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        # num_outs=5,
        # relu_before_extra_convs=True,
        norm_cfg=dict(type='BN', requires_grad=True, eps=1e-3, momentum=0.01)
    ),
)
# --------------------------------------------------------- #
