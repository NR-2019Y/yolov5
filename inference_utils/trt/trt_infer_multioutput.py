import tensorrt as trt
import numpy as np
from cuda import cudart
from collections import namedtuple
from .trt_infer import CHECK
from typing import Dict

DataInfo = namedtuple("DataInfo", ['host_data', 'cuda_data', 'shape'])


class TrtDetectorMultiOutput:
    def __init__(self, engine_file: str, use_async: bool = True):
        self._free = False
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        assert self.runtime
        with open(engine_file, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        if use_async:
            self.stream = CHECK(cudart.cudaStreamCreate())
        else:
            self.stream = None
        self.bindings = []
        self.data_output_info = dict()
        for name in self.engine:
            shape = self.engine.get_tensor_shape(name)
            size: int = trt.volume(shape)
            trt_type = self.engine.get_tensor_dtype(name)
            cuda_data = CHECK(cudart.cudaMalloc(size * trt_type.itemsize))
            self.bindings.append(cuda_data)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_shape = shape
                self.data_input_info = DataInfo(host_data=None, cuda_data=cuda_data, shape=shape)
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                np_dtype = trt.nptype(trt_type)
                host_data: np.ndarray = np.empty(size, dtype=np_dtype)
                self.data_output_info[name] = DataInfo(host_data=host_data, cuda_data=cuda_data, shape=shape)
        assert hasattr(self, 'input_shape') and hasattr(self, 'data_input_info')

    def _infer_async(self, input_blob: np.ndarray):
        cudart.cudaMemcpyAsync(self.data_input_info.cuda_data, input_blob.ctypes.data, input_blob.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream)
        for name, data_info in self.data_output_info.items():
            cudart.cudaMemcpyAsync(data_info.host_data.ctypes.data, data_info.cuda_data, data_info.host_data.nbytes,
                                   cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

    def _infer(self, input_blob: np.ndarray):
        cudart.cudaMemcpy(self.data_input_info.cuda_data, input_blob.ctypes.data, input_blob.nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self.context.execute_v2(self.bindings)
        for name, data_info in self.data_output_info.items():
            cudart.cudaMemcpy(data_info.host_data.ctypes.data, data_info.cuda_data, data_info.host_data.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    def infer(self, input_blob: np.ndarray) -> Dict[str, np.ndarray]:
        assert input_blob.data.contiguous
        if self.stream is None:
            self._infer(input_blob)
        else:
            self._infer_async(input_blob)
        outputs = dict()
        for name, data_info in self.data_output_info.items():
            outputs[name] = data_info.host_data.reshape(data_info.shape)
        return outputs

    def free(self):
        if not self._free:
            if self.stream is not None:
                cudart.cudaStreamDestroy(self.stream)
            cudart.cudaFree(self.data_input_info.cuda_data)
            for _, data_info in self.data_output_info.items():
                cudart.cudaFree(data_info.cuda_data)
            self._free = True

    def __del__(self):
        self.free()
