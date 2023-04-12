_base_ = [
    '../fcos_r50_fpn_gn-head_50e_ssdd.py'
]

work_dir = "E:/lrk/trail/logs/SAR/SSDD/modify/fpn/fcos_r50_nasfpn__gn-head_50e_ssdd"

# ------------------- FCOS + NASFPN -------------------------#
model = dict(
    neck=dict(
        _delete_=True,
        type='NASFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        stack_times=7,
        start_level=1,
        norm_cfg=dict(type='BN', requires_grad=True)),
)
# --------------------------------------------------------- #
