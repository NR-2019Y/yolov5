from typing import List
import pyppl.nn as pplnn
import pyppl.common as pplcommon


def _get_engines(res_types: List[str], quick_select: bool):
    engines = []
    for res_type in res_types:
        assert res_type in ('cuda', 'x86')
        if res_type == 'x86':
            options = pplnn.x86.EngineOptions()
            options.mm_policy = pplnn.x86.MM_COMPACT
            engine = pplnn.x86.EngineFactory.Create(options)
            assert engine
            engines.append(engine)
        elif res_type == 'cuda':
            options = pplnn.cuda.EngineOptions()
            options.device_id = 0
            options.mm_policy = pplnn.cuda.MM_BEST_FIT
            engine = pplnn.cuda.EngineFactory.Create(options)
            assert engine
            if quick_select:
                status = engine.Configure(pplnn.cuda.ENGINE_CONF_USE_DEFAULT_ALGORITHMS)
                assert status == pplcommon.RC_SUCCESS
            engines.append(engine)
    return engines


def create_runtime_from_onnx(onnx_file: str, res_types: List[str], quick_select: bool = False):
    engines = _get_engines(res_types, quick_select=quick_select)
    runtime_builder = pplnn.onnx.RuntimeBuilderFactory.Create()
    assert runtime_builder
    status = runtime_builder.LoadModelFromFile(onnx_file)
    assert status == pplcommon.RC_SUCCESS
    resources = pplnn.onnx.RuntimeBuilderResources()
    resources.engines = engines
    status = runtime_builder.SetResources(resources)
    assert status == pplcommon.RC_SUCCESS
    status = runtime_builder.Preprocess()
    assert status == pplcommon.RC_SUCCESS

    runtime = runtime_builder.CreateRuntime()
    assert runtime
    return runtime
