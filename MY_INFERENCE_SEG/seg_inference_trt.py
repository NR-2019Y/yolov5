import numpy as np
import time
import cv2
import os
import argparse
import yaml
from inference_utils.trt.trt_infer_multioutput import TrtDetectorMultiOutput
from MY_UTILS.SEG_TOOLS import process_mask
from MY_UTILS.NMS_NUMPY import NMS
from MY_UTILS.config import ROOT


def get_args():
    xargparser = argparse.ArgumentParser()
    xargparser.add_argument('--cfg', type=str,
                            default=os.path.join(ROOT, 'MY_INFERENCE_SEG/config/seg_smoke_config.yaml'))
    xargparser.add_argument('--engine', type=str,
                            default=os.path.join(ROOT, 'weights/smoke/r18y5s.engine'))
    xargparser.add_argument('--input-video', type=str, default=os.path.join(ROOT, 'VIDEO/1.mp4'))
    xargparser.add_argument('--alpha', type=float, default=0.5)
    args = xargparser.parse_args()

    with open(args.cfg, 'r', encoding='UTF-8') as f:
        cfg_dic = yaml.safe_load(f)

    if str.isdigit(args.input_video):
        args.input_video = int(args.input_video)

    args.max_dets = cfg_dic['max_dets']
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
    COLORS = args.colors
    MAX_DETS = args.max_dets
    CONF_THRESHOLD = args.conf_threshold
    IOU_THRESHOLD = args.iou_threshold
    # CLASS_NAMES = args.names
    VIDEO = cv2.VideoCapture(args.input_video)
    assert VIDEO.isOpened()
    np.random.seed(777)
    OBJ_COLORS = np.random.randint(0, 256, (MAX_DETS, 3)).tolist()
    ALPHA = args.alpha
    HEIGHT, WIDTH = args.height, args.width
    detector = TrtDetectorMultiOutput(args.engine)

    pause = False
    cv2.namedWindow('W', cv2.WINDOW_NORMAL)
    while True:
        if not pause:
            ret, ori_img = VIDEO.read()
            if not ret:
                print('finish!')
                break
            # img, ratio, (left, top) = letter_box(ori_img, (HEIGHT, WIDTH))

            ori_height, ori_width = ori_img.shape[:2]
            ratio = min(WIDTH / ori_width, HEIGHT / ori_height)
            new_height = round(ori_height * ratio)
            new_width = round(ori_width * ratio)
            left = (WIDTH - new_width) // 2
            top = (HEIGHT - new_height) // 2
            img_pad = np.full((HEIGHT, WIDTH, 3), 114, dtype=np.uint8)
            img_pad[top:top + new_height, left:left + new_width] = cv2.resize(ori_img, (new_width, new_height))

            tic = time.time()

            imgblob = np.ascontiguousarray(np.transpose(np.float32(img_pad[None, ..., ::-1]) / 255., (0, 3, 1, 2)))
            outputs = detector.infer(imgblob)
            pred = outputs['pred'][0]
            proto = outputs['proto'][0]
            toc = time.time()
            print(f'inference time: {toc - tic:.4f}')
            # reference: segment/predict.py

            obj_dets = NMS(pred, CONF_THRESHOLD, IOU_THRESHOLD, nm=32, top_k=MAX_DETS)
            if len(obj_dets) == 0:
                imm = ori_img
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
                    FONT_SCALE = 1.2
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

    VIDEO.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
