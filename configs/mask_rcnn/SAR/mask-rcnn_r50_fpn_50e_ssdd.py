_base_ = [
    '../../_base_/models/mask-rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance.py',
    '../../_base_/schedules/schedule_50e.py',
    '../../_base_/default_runtime.py'
]

work_dir = "E:/lrk/trail/logs/SAR/SSDD/baseline_hbb/mask-rcnn_hbox_r50_fpn_50e_ssdd"