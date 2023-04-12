_base_ = [
    '../fcos_r50_fpn_gn-head_50e_ssdd.py'
]

work_dir = "E:/lrk/trail/logs/SAR/SSDD/modify/fpn/fcos_r50_pafpn_gn-head_50e_ssdd"



# ------------------- Fcos + PAFPN -------------------------#
model = dict(
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
)
# --------------------------------------------------------- #




