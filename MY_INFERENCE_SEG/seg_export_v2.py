import io
import onnx
import onnxsim
import torch
import yaml
import os
from MY_UTILS.config import ROOT
from models.experimental import attempt_load
from models.yolo import Segment
import argparse

xargparser = argparse.ArgumentParser()
xargparser.add_argument('-i', '--ckpt', type=str)
xargparser.add_argument('-o', '--onnx', type=str)
args = xargparser.parse_args()

INPUT_SIZE = 640
CKPT_FILE = args.ckpt
ONNX_FILE = args.onnx

model = attempt_load(CKPT_FILE, inplace=True, fuse=True)
model.eval()
assert isinstance(model.model[-1], Segment)


def get_anchors(atype: str = 'anchors_p5_640'):
    anchors_files = os.path.join(ROOT, 'models/hub/anchors.yaml')
    with open(anchors_files, 'r', encoding='UTF-8') as f:
        dat = yaml.safe_load(f)
    return dat[atype]


ANCHORS = torch.tensor(get_anchors(), dtype=torch.float32).reshape(3, -1, 2)  # [3, 3, 2]
STRIDES = [8, 16, 32]

GRIDS = []
ANCHOR_GRIDS = []
for anchor, stride in zip(ANCHORS, STRIDES):
    s = INPUT_SIZE // stride
    rg = torch.arange(s, dtype=torch.float32)
    yv, xv = torch.meshgrid(rg, rg, indexing='ij')
    # [S, S, 2] -> [1, 1, S * S, 2]
    grid = torch.stack((xv, yv), dim=2).reshape(1, 1, -1, 2) - 0.5
    anchor = anchor.reshape(1, 3, 1, 2)
    GRIDS.append(grid.contiguous())
    ANCHOR_GRIDS.append(anchor.contiguous())


def new_forward(self, x: torch.Tensor):
    p = self.proto(x[0])
    assert self.nl == 3

    outputs = []

    for i in range(self.nl):
        x[i] = self.m[i](x[i])
        b, _, ny, nx = x[i].shape
        x[i] = x[i].reshape(b, self.na, self.no, ny * nx)
        x[i] = x[i].permute(0, 1, 3, 2)

        # xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), -1)
        xi = x[i]
        xy = xi[..., 0:2].sigmoid()
        wh = xi[..., 2:4].sigmoid()
        conf = xi[..., 4:5 + self.nc].sigmoid()
        mask = xi[..., 5 + self.nc:]

        xy = xy * torch.tensor(2. * STRIDES[i], dtype=torch.float32).reshape(1, 1, 1, 1)
        xy = xy + (GRIDS[i] * STRIDES[i])
        wh = wh.square() * (ANCHOR_GRIDS[i] * 4)
        curr_output = torch.cat((xy, wh, conf, mask), -1)
        curr_output = curr_output.reshape(b, -1, x[i].shape[-1])

        outputs.append(curr_output)

    return torch.cat(outputs, 1), p


model.model[-1].__class__.forward = new_forward

x = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
for _ in range(2):
    model(x)
with io.BytesIO() as f:
    torch.onnx.export(model, x, f, verbose=False,
                      do_constant_folding=True, opset_version=12,
                      input_names=['images'],
                      output_names=['pred', 'proto'])
    f.seek(0)
    onnx_model = onnx.load(f)

onnx.checker.check_model(onnx_model)
onnx_model, check = onnxsim.simplify(onnx_model)
assert check, 'check onnx simplify failed'
onnx.save(onnx_model, ONNX_FILE)
