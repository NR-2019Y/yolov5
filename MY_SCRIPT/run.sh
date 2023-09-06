# SERVER
python train.py --weights '' --cfg models/resnet34_yolov5s.yaml --data MY_DATA/gas_data_server.yaml --epochs 400 --batch-size 64 --imgsz 640 --workers 8 --name RUN_GAS_R34 --proj /mnt/b1/YOLOV5 &>_LOG/_RUN_GAS_R34
python train.py --weights '' --cfg models/resnet18_yolov5s.yaml --data MY_DATA/gas_data_server.yaml --epochs 400 --batch-size 64 --imgsz 640 --workers 8 --name RUN_GAS_R18 --proj /mnt/b1/YOLOV5 &>_LOG/_RUN_GAS_R18

python detect.py --weights runs/train/RUN_GAS_R18/weights/best.pt --source 0 --view-img

# SERVER
nohup python train.py --weights '' --cfg models/shufflenet_v2x20_yolov5s_w1.yaml --data MY_DATA/gas_data_server.yaml --epochs 400 --batch-size 32 --imgsz 640 --workers 8 --name RUN_GAS_shufflenet_v2x20_yolov5s_w1 --proj /mnt/b1/YOLOV5 &>_LOG/_RUN_GAS_shufflenet_v2x20_yolov5s_w1 &
