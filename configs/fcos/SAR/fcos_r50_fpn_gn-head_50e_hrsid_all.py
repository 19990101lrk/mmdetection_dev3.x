_base_ = [
    './fcos_r50_fpn_gn-head_50e_hrsid.py'
]

work_dir = "E:/lrk/trail/logs/SAR/HRSID/baseline/modify_fcos_r50_fpn_50e_hrsid_v1"

# -------------------------- MyFPN + Head  ------------------------------------- #
model = dict(
    neck=dict(
        type='MyFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        lateral_kernal_size=(5,),
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=True,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='CIoULoss', loss_weight=12.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
)

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    # dict(
    #     type='RandomChoiceResize',
    #     scales=[(608, 608), (800, 800)],
    #     keep_ratio=True),
    dict(type='RandomFlip', prob=[0.25, 0.25, 0.25], direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline
    )
)

# -------------------------------------------------------------------------------------------- #
