from pathlib import Path

import torch

# from models.tcn import TCNWithTwoInputs
from tcn import TCN

OUTPUT_DIR = Path('output')
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()

# ONNX_OUTPUT_PATH = 'output/tcn-with-two-inputs.onnx'
ONNX_OUTPUT_PATH = 'output/tcn.onnx'

# MODEL = TCNWithTwoInputs(
MODEL = TCN(
    **{
        # 'input_size': 243,
        'input_size': 233,

        # 'num_words': 10000,
        'output_size': 1,
        'num_channels': [32, 32, 32, 32],
        'kernel_size': 4,
        'dropout': 0.25,
    }
)

DEVICE = torch.device('cuda')

# INPUT_TENSORS = [torch.rand(1, 16, 233).to(DEVICE), torch.randint(low=1, high=10000, size=(1, 1, 10)).to(DEVICE)]
INPUT_TENSORS = [torch.rand(1, 16, 233).to(DEVICE)]

# INPUT_NAMES = ['input1', 'input2']
INPUT_NAMES = ['input']

OUTPUT_NAMES = ['output']

# DYNAMIC_AXES = {'input1': {0: 'batch_size'}, 'input2': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
DYNAMIC_AXES = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}


def main():
    MODEL.to(DEVICE)
    MODEL.eval()

    with torch.no_grad():
        torch.onnx.export(
            MODEL, tuple(INPUT_TENSORS), ONNX_OUTPUT_PATH, export_params=True, opset_version=11,
            do_constant_folding=True, input_names=INPUT_NAMES, output_names=OUTPUT_NAMES, dynamic_axes=DYNAMIC_AXES
        )


if __name__ == '__main__':
    main()
