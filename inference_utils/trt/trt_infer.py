import tensorrt as trt
import numpy as np
from cuda import cudart


def CHECK(args):
    err, result = args
    assert isinstance(err, cudart.cudaError_t)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Cuda Runtime Error: {err}")
    return result


class SimpleTrtDetector:
    def __init__(self, engine_file: str, use_async: bool = True):
        self._free = False
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        assert self.runtime
        with open(engine_file, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context
        if use_async:
            self.stream = CHECK(cudart.cudaStreamCreate())
        else:
            self.stream = None
        assert len(self.engine) == 2
        self.input_name, self.output_name = self.engine
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        input_size = trt.volume(self.input_shape)
        input_trt_dtype = self.engine.get_tensor_dtype(self.input_name)
        # input_dtype = trt.nptype(input_trt_dtype)
        # self.input_blob = np.empty(input_size, dtype=input_dtype)
        self.input_dtype = trt.nptype(input_trt_dtype)
        self.input_cuda_mem = CHECK(cudart.cudaMalloc(input_size * input_trt_dtype.itemsize))
        output_size = trt.volume(self.output_shape)
        output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        self.output_blob = np.empty(output_size, dtype=output_dtype)
        self.output_cuda_mem = CHECK(cudart.cudaMalloc(self.output_blob.nbytes))
        self.bindings = [self.input_cuda_mem, self.output_cuda_mem]

    def _infer(self, input_blob: np.ndarray):
        cudart.cudaMemcpy(self.input_cuda_mem, input_blob.ctypes.data, input_blob.nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self.context.execute_v2(self.bindings)
        cudart.cudaMemcpy(self.output_blob.ctypes.data, self.output_cuda_mem, self.output_blob.nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    def summary_info(self):
        d = dict()
        d[self.input_name] = self.input_shape
        d[self.output_name] = self.output_shape
        return d

    def _infer_async(self, input_blob: np.ndarray):
        cudart.cudaMemcpyAsync(self.input_cuda_mem, input_blob.ctypes.data, input_blob.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream)
        cudart.cudaMemcpyAsync(self.output_blob.ctypes.data, self.output_cuda_mem, self.output_blob.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        assert input_blob.data.contiguous
        assert input_blob.dtype == self.input_dtype
        if self.stream is None:
            self._infer(input_blob)
        else:
            self._infer_async(input_blob)
        return self.output_blob.reshape(self.output_shape)

    def free(self):
        if not self._free:
            if self.stream is not None:
                cudart.cudaStreamDestroy(self.stream)
            cudart.cudaFree(self.input_cuda_mem)
            cudart.cudaFree(self.output_cuda_mem)
            self._free = True

    def __del__(self):
        self.free()
