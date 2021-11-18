import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxoptimizer
import onnxsim
from onnx.checker import check_model
from onnx.onnx_ml_pb2 import _TENSORPROTO_DATATYPE
from polygraphy import util
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import (
    TrtRunner,
    CreateConfig,
    Calibrator,
    Profile,
    save_engine,
    engine_from_network,
    engine_from_bytes,
    network_from_onnx_path,
)
from polygraphy.comparator import Comparator
from polygraphy.logger import G_LOGGER
from scipy.stats import pearsonr

G_LOGGER.severity = G_LOGGER.INFO

MODEL_NAME = 'tcn'
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
            use_onnx_optimizer=True,
            use_onnx_simplifier=False,
            min_batch_size=1,
            max_batch_size=1,
            opt_batch_size=1,
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
            use_onnx_optimizer (bool):
                    Whether to use ONNX Optimizer. For more details: https://github.com/onnx/optimizer
            use_onnx_simplifier (bool):
                    Whether to use ONNX Simplifier. For more details: https://github.com/daquexian/onnx-simplifier
            min_batch_size (int):
                    The minimal batch size to be used in model inferences. Only works for models whose inputs have
                    dynamic shapes.
            max_batch_size (int):
                    The maximum batch size to be used in model inferences. Only works for models whose inputs have
                    dynamic shapes.
            opt_batch_size (int):
                    The batch size to be mainly used and optimized in model inferences. Only works for models whose
                    inputs have dynamic shapes.
            calibration_data_generator (generator):
                    A generator to provide data for int8 calibration. The data should be in the following format:
                        {'input_name_1': np.ndarray(), 'input_name_2': np.ndarray()}
                    '_random_data_generator' should be a good sample to start writing your own data generator.
                    If unspecified in the int8 mode, random data will be generated for calibration.
            calibration_cache (Union[str, file-like]):
                    A path or file-like object where we load/save the int8 calibration cache to speed up the
                    building process.
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

        self.use_onnx_optimizer_ = use_onnx_optimizer
        self.use_onnx_simplifier_ = use_onnx_simplifier
        self.min_batch_size_ = min_batch_size
        self.max_batch_size_ = max_batch_size
        self.opt_batch_size_ = opt_batch_size

        self.int8 = util.default(int8, False)
        self.calibration_data_generator_ = calibration_data_generator
        self.calibration_cache_ = calibration_cache

        # {'input_name': ([dim0, dim1, ...], 'dtype', is_dynamic_batch_size)}
        self.inputs_: Dict[str, Tuple[List[int], str, bool]] = {}
        self.outputs_: List[str] = []

    def run(self, onnx_model_path, output_engine='out.trt', optimized_model_path='optimized.onnx',
            remove_optimized_model=True, check_accuracy=True, override=True):
        """
        Convert an ONNX model to a TensorRT engine and check output accuracy of the converted model.

        Args:
            onnx_model_path (Union[str, file-like]):
                    The ONNX model to be converted
            output_engine (Union[str, file-like]):
                    The converted TensorRT engine
            optimized_model_path (Union[str, file-like]):
                    The name of temporary optimized model
            remove_optimized_model (bool):
                    Whether to remove the temporary optimized model
            check_accuracy (bool):
                    Whether to compare between the accuracy of the converted model and the original model
            override (bool):
                    Whether to override the converted model if it already exists
        """
        model = onnx.load(onnx_model_path)
        check_model(model)
        self._set_inputs_and_outputs(model)

        if not override and Path(output_engine).exists():
            G_LOGGER.warning(f"'{output_engine}' already exists, "
                             f"skip building because 'override' is set to 'False'")
            if check_accuracy:
                with open(output_engine, 'rb') as f:
                    engine = engine_from_bytes(f.read())
                    self.check_accuracy(engine, onnx_model_path)
            return

        # Optimize ONNX model
        if self.use_onnx_optimizer_ or self.use_onnx_simplifier_:
            if self.use_onnx_optimizer_:
                model = self._run_onnx_optimizer(model)
            if self.use_onnx_simplifier_:
                if any(x[2] for x in self.inputs_.values()):
                    # There are dynamic input shapes
                    model = self._run_onnx_simplifier(model, dynamic_input_shape=True,
                                                      input_shapes=self._get_fixed_input_shapes())
                else:
                    model = self._run_onnx_simplifier(model)
            onnx.save(model, optimized_model_path)

            network = network_from_onnx_path(optimized_model_path)

            if remove_optimized_model:
                os.remove(optimized_model_path)
        else:
            network = network_from_onnx_path(onnx_model_path)

        # Set optimization profiles for dynamic shapes
        profile = self._set_optimization_profiles()

        # Set calibrator for Int8 quantization
        if self.int8 and self.calibration_data_generator_ is None:
            self.calibration_data_generator_ = Calibrator(data_loader=self._random_data_generator(),
                                                          cache=self.calibration_cache_)

        config = CreateConfig(
            profiles=profile,
            calibrator=self.calibration_data_generator_,
            **self._get_tensorrt_config()
        )
        engine = engine_from_network(network, config=config)
        save_engine(engine, output_engine)

        # Compare accuracy
        if check_accuracy:
            self.check_accuracy(engine, onnx_model_path)

    def check_accuracy(self, engine, onnx_model_path):
        build_onnxrt_session = SessionFromOnnx(onnx_model_path)
        runners = [
            OnnxrtRunner(build_onnxrt_session),
            TrtRunner(engine)
        ]

        run_results = Comparator.run(runners, data_loader=self._random_data_generator(1))
        for output_name in self.outputs_:
            result_arrays = [list(run_results.values())[i][0].dct[output_name].arr for i in range(len(list(runners)))]
            G_LOGGER.info('-' * 10 + f"Pearson correlation coefficient for output '{output_name}': "
                                     f'{pearsonr(result_arrays[0].ravel(), result_arrays[1].ravel())}' + '-' * 10)
        Comparator.compare_accuracy(run_results)

    def _set_inputs_and_outputs(self, onnx_model):
        for input_ in onnx_model.graph.input:
            name, raw_dtype = input_.name, input_.type.tensor_type

            dtype = _TENSORPROTO_DATATYPE.values_by_number[raw_dtype.elem_type].name.lower()

            if raw_dtype.HasField("shape"):
                shape_list = []
                for i, d in enumerate(raw_dtype.shape.dim):
                    if d.HasField("dim_value"):
                        shape_list.append(d.dim_value)
                    elif d.HasField("dim_param"):
                        if i > 0:
                            raise TypeError(f'Only the batch dimension can be dynamic, '
                                            f"while the {i}-th dimension of '{name}' is dynamic")
                        shape_list.append(-1)
                    else:
                        raise TypeError(f"The {i}-th dimension of input '{name}' is invalid")
                self.inputs_[name] = shape_list, dtype, shape_list[0] == -1
            else:
                raise ValueError(f"The input '{name}' of ONNX does not have a shape field")

        for output in onnx_model.graph.output:
            self.outputs_.append(output.name)

    def _get_tensorrt_config(self):
        return {k: v for k, v in self.__dict__.items() if not k.endswith('_')}

    def _set_optimization_profiles(self):
        profile = Profile()
        for name, (shape_list, _, is_dynamic_batch_size) in self.inputs_.items():
            if is_dynamic_batch_size:
                profile.add(
                    name,
                    min=tuple([self.min_batch_size_] + shape_list[1:]),
                    max=tuple([self.max_batch_size_] + shape_list[1:]),
                    opt=tuple([self.opt_batch_size_] + shape_list[1:])
                )

        return [profile] if len(profile) > 0 else None

    def _random_data_generator(self, num_iterations=4):
        for _ in range(num_iterations):
            data = {}
            for name, (shape, dtype, is_dynamic_batch_size) in self.inputs_.items():
                if is_dynamic_batch_size:
                    shape = [self.opt_batch_size_] + shape[1:]

                if 'float' in dtype:
                    data[name] = np.random.random(size=shape).astype('float32')
                elif 'int' in dtype:
                    data[name] = np.random.randint(low=1, high=4, size=shape)
                else:
                    raise ValueError('Only float or int inputs are supported for calibration using random data')

            yield data

    def _get_fixed_input_shapes(self):
        shapes = {}
        for name, (shape_list, _, is_dynamic_batch_size) in self.inputs_.items():
            if is_dynamic_batch_size:
                shapes[name] = [1] + shape_list[1:]
            else:
                shapes[name] = shape_list

        return shapes

    @staticmethod
    def _run_onnx_optimizer(onnx_model):
        model = onnxoptimizer.optimize(onnx_model)
        check_model(model)
        return model

    @staticmethod
    def _run_onnx_simplifier(onnx_model, **kwargs):
        model, success = onnxsim.simplify(onnx_model, **kwargs)
        assert success, "Simplified ONNX model could not be validated"
        return model


def main():
    trt_converter = TRTConverter(
        max_workspace_size=MAX_WORKSPACE_SIZE,
        min_batch_size=BATCH_SIZE,
        max_batch_size=BATCH_SIZE,
        opt_batch_size=BATCH_SIZE,
        int8=INT8,
        calibration_cache=OUTPUT_CACHE_PATH
    )
    trt_converter.run(INPUT_ONNX_MODEL_PATH, OUTPUT_TENSORRT_MODEL_PATH, override=False)


if __name__ == "__main__":
    main()
