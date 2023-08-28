import io
import onnx
import onnxsim
import torch
from models.experimental import attempt_load
from models.yolo import Segment
import argparse

xargparser = argparse.ArgumentParser()
xargparser.add_argument('-i', '--ckpt', type=str)
xargparser.add_argument('-o', '--onnx', type=str)
args = xargparser.parse_args()

CKPT_FILE = args.ckpt
ONNX_FILE = args.onnx

model = attempt_load(CKPT_FILE, inplace=True, fuse=True)
model.eval()
assert isinstance(model.model[-1], Segment)


def new_forward(self, x):
    p = self.proto(x[0])
    for i in range(self.nl):
        x[i] = self.m[i](x[i])
        b, _, ny, nx = x[i].shape
        x[i] = x[i].reshape(b, self.na, self.no, ny * nx)
        x[i] = x[i].permute(0, 1, 3, 2)
    return x[0], x[1], x[2], p


model.model[-1].__class__.forward = new_forward

x = torch.rand(1, 3, 640, 640)
for _ in range(2):
    model(x)
with io.BytesIO() as f:
    torch.onnx.export(model, x, f, verbose=False,
                      do_constant_folding=True, opset_version=12,
                      input_names=['images'],
                      output_names=['pred0', 'pred1', 'pred2', 'proto'])
    f.seek(0)
    onnx_model = onnx.load(f)

onnx.checker.check_model(onnx_model)
onnx_model, check = onnxsim.simplify(onnx_model)
assert check, 'check onnx simplify failed'
onnx.save(onnx_model, ONNX_FILE)
