_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_50e.py',
    '../../_base_/default_runtime.py',
    '.././retinanet_tta.py'
]

work_dir = "E:/lrk/trail/logs/SAR/SSDD/baseline_hbb/retinanet_hbox_r50_fpn_50e_ssdd"



