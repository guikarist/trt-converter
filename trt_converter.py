from typing import Dict, List, Tuple

import numpy as np
from polygraphy import util
from polygraphy.backend.trt import (
    CreateConfig,
    Calibrator,
    Profile,
    save_engine,
    engine_from_network,
    network_from_onnx_path,
)
from polygraphy.logger import G_LOGGER

G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE

MODEL_NAME = 'tcn-with-two-inputs'
BATCH_SIZE = 4500
INT8 = True
INT8_STRING = INT8 * '-int8'
MAX_WORKSPACE_SIZE = 1024 * 1024 * 1024

INPUT_ONNX_MODEL_PATH = f'output/{MODEL_NAME}.onnx'
OUTPUT_CACHE_PATH = f'output/{MODEL_NAME}-{BATCH_SIZE}{INT8_STRING}.cache'
OUTPUT_TENSORRT_MODEL_PATH = f'output/{MODEL_NAME}-{BATCH_SIZE}{INT8_STRING}.trt'


class TRTConverter:
    def __init__(
            self,
            max_workspace_size=None,
            tf32=None,
            fp16=None,
            int8=None,
            obey_precision_constraints=None,
            load_timing_cache=None,
            algorithm_selector=None,
            sparse_weights=None,
            tactic_sources=None,
            restricted=None,
            use_dla=None,
            allow_gpu_fallback=None,
            min_batch_size=1,
            opt_batch_size=1,
            max_batch_size=1,
            calibration_data_generator=None,
            calibration_cache=None,
    ):
        """
        Creates a TensorRT IBuilderConfig that can be used by EngineFromNetwork.

        Args:
            max_workspace_size (int):
                    The maximum workspace size, in bytes, when building the engine.
                    Defaults to 16 MiB.
            tf32 (bool):
                    Whether to build the engine with TF32 precision enabled.
                    Defaults to False.
            fp16 (bool):
                    Whether to build the engine with FP16 precision enabled.
                    Defaults to False.
            int8 (bool):
                    Whether to build the engine with INT8 precision enabled.
                    Defaults to False.
            obey_precision_constraints (bool):
                    If True, require that layers execute in specified precisions.
                    Defaults to False.
            load_timing_cache (Union[str, file-like]):
                    A path or file-like object from which to load a tactic timing cache.
                    Providing a tactic timing cache can speed up the engine building process.
                    Caches can be generated while building an engine.
            algorithm_selector (trt.IAlgorithmSelector):
                    An algorithm selector. Allows the user to control how tactics are selected
                    instead of letting TensorRT select them automatically.
            sparse_weights (bool):
                    Whether to enable optimizations for sparse weights.
                    Defaults to False.
            tactic_sources (List[trt.TacticSource]):
                    The tactic sources to enable. This controls which libraries (e.g. cudnn, cublas, etc.)
                    TensorRT is allowed to load tactics from.
                    Use an empty list to disable all tactic sources.
                    Defaults to TensorRT's default tactic sources.
            restricted (bool):
                    Whether to enable safety scope checking in the builder. This will check if the network
                    and builder configuration are compatible with safety scope.
                    Defaults to False.
            use_dla (bool):
                    [EXPERIMENTAL] Whether to enable DLA as the default device type.
                    Defaults to False.
            allow_gpu_fallback (bool):
                    [EXPERIMENTAL] When DLA is enabled, whether to allow layers to fall back to GPU if they cannot be
                    run on DLA.
                    Has no effect if DLA is not enabled.
                    Defaults to False.
            calibration_data_generator (generator):
                    A generator to provide data for int8 calibration. The data should be in the following format:
                        {'input_name_1': np.ndarray(), 'input_name_2': np.ndarray()}
                    '_random_data_generator' should be a good sample to start writing your own data generator.
                    If unspecified in the int8 mode, random data will be generated for calibration.
            calibration_cache (Union[str, file-like]):
                    A path or file-like object where we load/save the int8 calibration cache to speed up the
                    building process.
            min_batch_size (int):
                    The minimal batch size to be used in model inferences. Only works for models whose inputs have
                    dynamic shapes.
            max_batch_size (int):
                    The maximum batch size to be used in model inferences. Only works for models whose inputs have
                    dynamic shapes.
            opt_batch_size (int):
                    The batch size to be mainly used and optimized in model inferences. Only works for models whose
                    inputs have dynamic shapes.
        """
        self.max_workspace_size = util.default(max_workspace_size, 1 << 24)
        self.tf32 = util.default(tf32, False)
        self.fp16 = util.default(fp16, False)
        self.obey_precision_constraints = util.default(obey_precision_constraints, False)
        self.restricted = util.default(restricted, False)
        self.load_timing_cache = load_timing_cache
        self.algorithm_selector = algorithm_selector
        self.sparse_weights = util.default(sparse_weights, False)
        self.tactic_sources = tactic_sources
        self.use_dla = util.default(use_dla, False)
        self.allow_gpu_fallback = util.default(allow_gpu_fallback, False)

        self.min_batch_size_ = min_batch_size
        self.max_batch_size_ = max_batch_size
        self.opt_batch_size_ = opt_batch_size

        self.int8 = util.default(int8, False)
        self.calibration_data_generator_ = calibration_data_generator
        self.calibration_cache_ = calibration_cache

        self.inputs_: Dict[str, Tuple[List[int], str]] = {}  # {'input_name': ([dim0, dim1, ...], 'dtype')}

    def run(self, onnx_model, output_engine='out.trt'):
        """
        Convert an ONNX model to a TensorRT engine.

        Args:
            onnx_model (Union[str, file-like]):
                    The ONNX model to be converted
            output_engine (Union[str, file-like]):
                    The converted TensorRT engine
        """
        network = network_from_onnx_path(onnx_model)

        profile = self._set_optimization_profiles(network[1])
        if self.int8:
            if self.calibration_data_generator_ is None:
                self.calibration_data_generator_ = Calibrator(data_loader=self._random_data_generator(),
                                                              cache=self.calibration_cache_)
        config = CreateConfig(profiles=profile,
                              calibrator=self.calibration_data_generator_,
                              **self._get_tensorrt_config())

        engine = engine_from_network(network, config=config)
        save_engine(engine, output_engine)

    def _get_tensorrt_config(self):
        return {k: v for k, v in self.__dict__.items() if not k.endswith('_')}

    def _set_optimization_profiles(self, network):
        profile = Profile()
        for i in range(network.num_inputs):
            input_ = network.get_input(i)
            name, shape, dtype = input_.name, input_.shape, input_.dtype.name.lower()
            shape = network.get_input(i).shape

            shape_list = []
            found_dynamic = False
            for dim_i, dim in enumerate(shape):
                if dim < 0:
                    if dim_i > 0:
                        raise NotImplementedError(f'Only the batch dimension can be dynamic, '
                                                  f'while the {dim_i}-th dimension of {name} is dynamic')
                    found_dynamic = True

                shape_list.append(dim)

            if found_dynamic:
                profile.add(
                    name,
                    min=tuple([self.min_batch_size_] + shape_list[1:]),
                    max=tuple([self.max_batch_size_] + shape_list[1:]),
                    opt=tuple([self.opt_batch_size_] + shape_list[1:])
                )
                self.inputs_[name] = [self.opt_batch_size_] + shape_list[1:], dtype
            else:
                self.inputs_[name] = shape_list, dtype

        return [profile] if len(profile) > 0 else None

    def _random_data_generator(self):
        for _ in range(4):
            data = {}
            for name, (shape, dtype) in self.inputs_.items():
                if 'float' in dtype:
                    data[name] = np.random.random(size=shape).astype('float32')
                elif 'int' in dtype:
                    data[name] = np.random.randint(low=1, high=4, size=shape)
                else:
                    raise ValueError('Only float or int inputs are supported for calibration using random data')

            yield data


def main():
    trt_converter = TRTConverter(
        max_workspace_size=MAX_WORKSPACE_SIZE,
        min_batch_size=BATCH_SIZE,
        max_batch_size=BATCH_SIZE,
        opt_batch_size=BATCH_SIZE,
        int8=INT8,
        calibration_cache=OUTPUT_CACHE_PATH
    )
    trt_converter.run(INPUT_ONNX_MODEL_PATH, OUTPUT_TENSORRT_MODEL_PATH)


if __name__ == "__main__":
    main()
