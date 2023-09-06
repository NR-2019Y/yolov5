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


ln -s $PWD/runs/train-seg/r34y5s_smoke_seg/weights/best.pt weights/smoke/r34y5s.pt
ln -s $PWD/runs/train-seg/r18y5s_smoke_seg/weights/best.pt weights/smoke/r18y5s.pt
ln -s $PWD/runs/train-seg/shuv2x20y5s_smoke_seg/weights/best.pt weights/smoke/shuv2x20y5s.pt
ln -s $PWD/runs_old/train-seg/coco-smoke-5s-seg/weights/best.pt weights/smoke/y5s.pt
ln -s $PWD/runs_old/train-seg/coco-smoke-seg/weights/best.pt weights/smoke/y5n.pt

export PYTHONPATH=.
for f in weights/smoke/*.pt; do
  python MY_INFERENCE_SEG/seg_export_v2.py -i $f -o ${f%.pt}.onnx
done

for f in weights/smoke/*.onnx; do
  trtexec --onnx=$f --saveEngine=${f%.onnx}.engine
done

for f in weights/smoke/*.onnx; do
  PREFIX=${f%.onnx}
  onnx2ncnn $f ${PREFIX}.ori.param ${PREFIX}.ori.bin
  ncnnoptimize ${PREFIX}.ori.param ${PREFIX}.ori.bin ${PREFIX}.param ${PREFIX}.bin 0
  rm ${PREFIX}.ori.param ${PREFIX}.ori.bin
done
