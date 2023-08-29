import os.path
import time
import cv2
import numpy as np
import yaml
import argparse
from MY_UTILS.NMS_NUMPY import NMS
from MY_UTILS.SEG_TOOLS import process_mask
from MY_UTILS.config import ROOT
from inference_utils.pplnn.common import create_runtime_from_onnx
import pyppl.common as pplcommon


def get_args():
    xargparser = argparse.ArgumentParser()
    xargparser.add_argument('--cfg', type=str,
                            default=os.path.join(ROOT, 'MY_INFERENCE_SEG/config/seg_smoke_config.yaml'))
    xargparser.add_argument('--weights', type=str,
                            default=os.path.join(ROOT, 'runs_old/train-seg/coco-smoke-seg/weights/best.onnx'))
    # default=os.path.join(ROOT, 'weights/yolov5n-seg-smoke.onnx'))
    xargparser.add_argument('--input-video', type=str, default=os.path.join(ROOT, 'VIDEO/1.mp4'))
    xargparser.add_argument('--quick-select', type=int, default=1)
    xargparser.add_argument('--alpha', type=float, default=0.3)
    args = xargparser.parse_args()

    with open(args.cfg, 'r', encoding='UTF-8') as f:
        cfg_dic = yaml.safe_load(f)

    if str.isdigit(args.input_video):
        args.input_video = int(args.input_video)

    args.max_dets = cfg_dic['max_dets']
    args.nm = cfg_dic['nm']
    args.nc = cfg_dic['nc']
    args.names = cfg_dic['names']
    args.colors = cfg_dic['colors']
    args.width = cfg_dic['width']
    args.height = cfg_dic['height']
    args.conf_threshold = cfg_dic['conf_threshold']
    args.iou_threshold = cfg_dic['iou_threshold']
    return args


def main():
    args = get_args()
    ALPHA = args.alpha
    CONF_THRESHOLD = args.conf_threshold
    IOU_THRESHOLD = args.iou_threshold
    MAX_DETS = args.max_dets
    VIDEO_PATH = args.input_video
    HEIGHT = args.height
    WIDTH = args.width
    NUM_CLASSES = args.nc

    video = cv2.VideoCapture(VIDEO_PATH)
    assert video.isOpened()

    np.random.seed(777)
    COLORS = args.colors
    OBJ_COLORS = np.random.randint(0, 256, (MAX_DETS, 3)).tolist()

    runtime = create_runtime_from_onnx(args.weights, ['cuda'], quick_select=args.quick_select)
    tensor_input = runtime.GetInputTensor(0)

    cv2.namedWindow('W', cv2.WINDOW_NORMAL)
    pause = False
    while True:
        if not pause:
            ret, ori_image = video.read()
            if not ret:
                print('finish!')
                break
            tic = time.time()
            ori_height, ori_width = ori_image.shape[:2]
            ratio = min(WIDTH / ori_width, HEIGHT / ori_height)
            new_width = round(ori_width * ratio)
            new_height = round(ori_height * ratio)
            padw = WIDTH - new_width
            padh = HEIGHT - new_height
            left = padw // 2
            top = padh // 2

            img_pad = np.pad(cv2.resize(ori_image, (new_width, new_height)),
                             pad_width=[[top, padh - top], [left, padw - left], [0, 0]],
                             mode='constant', constant_values=0)
            imgblob = np.transpose(np.float32(img_pad[..., ::-1]) / 255., (2, 0, 1))
            imgblob = np.ascontiguousarray(imgblob[None])

            status = tensor_input.ConvertFromHost(imgblob)
            assert status == pplcommon.RC_SUCCESS

            runtime.Run()

            tensor_output0 = runtime.GetOutputTensor(0)
            tensor_output1 = runtime.GetOutputTensor(1)

            pred = np.array(tensor_output0.ConvertToHost(), copy=False)[0]
            proto = np.array(tensor_output1.ConvertToHost(), copy=False)[0]
            obj_dets = NMS(pred, CONF_THRESHOLD, IOU_THRESHOLD, nm=args.nm, top_k=MAX_DETS)
            toc = time.time()
            print(f'time: {(toc - tic) * 1000:.4f} ms')

            if len(obj_dets) == 0:
                imm = ori_image
            else:
                masks = process_mask(proto, obj_dets[:, 6:], obj_dets[:, :4], (HEIGHT, WIDTH))  # [n, h, w]
                obj_dets[:, [0, 2]] -= left
                obj_dets[:, [1, 3]] -= top
                obj_dets[:, :4] /= ratio

                masks = masks.astype(bool)
                img_mask = img_pad.copy()
                for i, mask in enumerate(masks[::-1]):
                    img_mask[mask] = OBJ_COLORS[i]
                imm = np.uint8(img_pad * (1 - ALPHA) + img_mask * ALPHA)
                imm = imm[top:top + new_height, left:left + new_width]
                imm = cv2.resize(imm, (ori_width, ori_height))
            for *xyxy, conf, cls in obj_dets[:, :6]:
                x1, y1, x2, y2 = map(round, xyxy)
                cls = int(cls)
                # text = f'{CLASS_NAMES[cls]}:{conf:.4f}'
                text = f'{conf:.4f}'
                FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
                FONT_SCALE = 0.8
                FONT_THICKNESS = 2
                (fw, fh), fb = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
                cv2.rectangle(imm, (x1, y1 - fh - fb), (x1 + fw, y1), COLORS[cls], -1)
                cv2.rectangle(imm, (x1, y1), (x2, y2), COLORS[cls], 2)
                cv2.putText(imm, text, (x1, y1 - fb), FONT_FACE, FONT_SCALE, [0, 0, 0], FONT_THICKNESS)

            cv2.imshow('W', imm)
        key = cv2.waitKey(5)
        if key == 27:
            print('exit')
            break
        if key == ord('p') or key == ord('P'):
            pause = not pause

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
