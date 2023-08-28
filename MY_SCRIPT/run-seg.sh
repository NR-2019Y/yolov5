# python segment/train.py --data MY_DATA/smoke_seg.yaml --cfg models/segment/resnet18_yolov5s-seg.yaml --weights '' --name r18y5s_smoke_seg \
# --batch-size 4 --workers 8 --epochs 2
# server
python segment/train.py --data MY_DATA/smoke_seg_server.yaml --cfg models/segment/resnet18_yolov5s-seg.yaml --weights '' --name r18y5s_smoke_seg \
 --proj /mnt/b1/YOLOV5_SEG --batch-size 32 --workers 8 --epochs 400 &> _LOG/_RUNSEG_r18y5s_smoke_seg
python segment/train.py --data MY_DATA/smoke_seg_server.yaml --cfg models/segment/resnet34_yolov5s-seg.yaml --weights '' --name r34y5s_smoke_seg \
 --proj /mnt/b1/YOLOV5_SEG --batch-size 32 --workers 8 --epochs 400 &> _LOG/_RUNSEG_r34y5s_smoke_seg
python segment/train.py --data MY_DATA/smoke_seg_server.yaml --cfg models/segment/shufflenet_v2x20_yolov5s_w1-seg.yaml \
--weights '' --name shuv2x20y5s_smoke_seg --proj /mnt/b1/YOLOV5_SEG --batch-size 32 --workers 8 --epochs 400 &>_LOG/_RUNSEG_shuv2x20y5s_smoke_seg

# LOCAL
PYTHONPATH=. python MY_INFERENCE_SEG/seg_inference_pt.py --weights runs/train-seg/r34y5s_smoke_seg/weights/best.pt
PYTHONPATH=. python MY_INFERENCE_SEG/seg_inference_pt.py --weights runs/train-seg/r18y5s_smoke_seg/weights/best.pt
PYTHONPATH=. python MY_INFERENCE_SEG/seg_inference_pt.py --weights runs/train-seg/shuv2x20y5s_smoke_seg/weights/best.pt
PYTHONPATH=. python MY_INFERENCE_SEG/seg_inference_pt.py --weights runs_old/train-seg/coco-smoke-5s-seg/weights/best.pt
PYTHONPATH=. python MY_INFERENCE_SEG/seg_inference_pt.py --weights runs_old/train-seg/coco-smoke-seg/weights/best.pt
