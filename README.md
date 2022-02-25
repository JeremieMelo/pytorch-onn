<p align="center">
<img src="torchonn_logo.jpg" alt="torchonn Logo" width="450">
</p>

```
@inproceedings{jiaqigu2021L2ight,
    title     = {L2ight: Enabling On-Chip Learning for Optical Neural Networks via Efficient in-situ Subspace Optimization},
    author    = {Jiaqi Gu and Hanqing Zhu and Chenghao Feng and Zixuan Jiang and Ray T. Chen and David Z. Pan},
    booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
    year      = {2021}
}
```

# Torch-ONN
A PyTorch-centric Optical Neural Network Library

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/JeremieMelo/pytorch-onn/blob/release/LICENSE)

<!-- toc -->

- [Torch-ONN](#torch-onn)
  - [News](#news)
  - [Installation](#installation)
    - [From Source](#from-source)
      - [Dependencies](#dependencies)
      - [Get the PyTorch-ONN Source](#get-the-pytorch-onn-source)
      - [Install PyTorch-ONN](#install-pytorch-onn)
  - [Usage](#usage)
  - [Features](#features)
  - [TODOs](#todos)
  - [Files](#files)
- [More Examples](#more-examples)
  - [Contact](#contact)
- [Related Projects using PyTorch-ONN Library](#related-projects-using-pytorch-onn-library)

<!-- tocstop -->
## News
- _**11/28/2021**_: v0.0.2 available. FFT-ONN-family is supported with trainable butterfly meshes for area-efficient frequency-domain optical neurocomputing! (Gu+, [FFT-ONN](https://doi.org/10.1109/ASP-DAC47756.2020.9045156) ASP-DAC 2020) (Gu+, [FFT-ONN-v2](https://doi.org/10.1109/TCAD.2020.3027649) IEEE TCAD 2021) (Feng+, [PSNN](https://arxiv.org/abs/2111.06705) arXiv 2021)
- _**06/10/2021**_: v0.0.1 available. MZI-ONN (Shen+, [MZI-ONN](https://doi.org/10.1038/nphoton.2017.93)) is supported. Feedbacks are highly welcomed!

## Installation

### From Source

#### Dependencies
- Python >= 3.6
- PyTorch >= 1.8.0
- Tensorflow-gpu >= 2.5.0
- pyutils >= 0.0.1
- Others are listed in requirements.txt
- GPU model training requires NVIDIA GPUs and compatible CUDA

#### Get the PyTorch-ONN Source
```bash
git clone https://github.com/JeremieMelo/pytorch-onn.git
```

#### Install PyTorch-ONN
```bash
cd pytorch-onn
python3 setup.py install --user clean
```
or
```bash
./setup.sh
```

## Usage
Construct optical NN models as simple as constructing a normal pytorch model.
```python
import torch.nn as nn
import torch.nn.functional as F
import torchonn as onn
from torchonn.models import ONNBaseModel

class ONNModel(ONNBaseModel):
    def __init__(self, device=torch.device("cuda:0)):
        super().__init__(device=device)
        self.conv = onn.layers.MZIBlockConv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            device=device,
        )
    self.pool = nn.AdaptiveAvgPool2d(5)
    self.linear = onn.layers.MZIBlockLinear(
        in_features=8*5*5,
        out_features=10,
        bias=True,
        miniblock=4,
        mode="usv",
        decompose_alg="clements",
        photodetect=True,
        device=device,
    )

    self.conv.reset_parameters()
    self.linear.reset_parameters()

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x
```

## Features
- Support pytorch training MZI-based ONNs. Support MZI-based Linear, Conv2d, BlockLinear, and BlockConv2d. Support `weight`, `usv`, `phase` modes and their conversion.
- Support phase **quantization** and **non-ideality injection**, including phase shifter gamma error, phase variations, and crosstalk.
- **CUDA-accelerated batched** MZI array decomposition and reconstruction for ultra-fast real/complex matrix mapping, which achieves 10-50X speedup over CPU-based unitary group parametrization. Francis (Triangle), Reck (Triangle), Clements (Rectangle) styles MZI meshes are supported. To see the efficiency of our CUDA implementation, try the following unittest command at root directory,
 `python3 unitest/test_op.py`
 , and check the runtime comparison.
- Support pytorch training general frequency-domain ONNs (Gu+, [FFT-ONN](https://doi.org/10.1109/ASP-DAC47756.2020.9045156) ASP-DAC 2020) (Gu+, [FFT-ONN-v2](https://doi.org/10.1109/TCAD.2020.3027649) IEEE TCAD 2021) (Feng+, [PSNN](https://arxiv.org/abs/2111.06705)). Support FFT-ONN BlockLinear, and BlockConv2d. Support `fft`, `hadamard`, `zero_bias`, and `trainable` modes.

## TODOs
- [ ] Support micro-ring resonator (MRR)-based ONN. (Tait+, [SciRep](https://doi.org/10.1038/s41598-017-07754-z) 2017)
<!-- - [x] Support general frequency-domain ONN. (Gu+, [FFT-ONN](https://doi.org/10.1109/ASP-DAC47756.2020.9045156) ASP-DAC 2020) (Gu+, [FFT-ONN-v2](https://doi.org/10.1109/TCAD.2020.3027649) IEEE TCAD 2021) -->
- [ ] Support multi-operand micro-ring (MORR)-based ONN. (Gu+, [SqueezeLight](https://jeremiemelo.github.io/publications/papers/ONN_DATE2021_SqueezeLight_Gu.pdf) DATE 2021)
- [ ] Support differentiable quantization-aware training. (Gu+, [ROQ](https://doi.org/10.23919/DATE48585.2020.9116521) DATE 2020)
- [ ] Support ONN on-chip learning via zeroth-order optimization. (Gu+, [FLOPS](https://doi.org/10.1109/DAC18072.2020.9218593) DAC 2020) (Gu+, [MixedTrain](https://arxiv.org/abs/2012.11148) AAAI 2021)


## Files
| File             | Description |
| ---------------- | ----------- |
| torchonn/        | Library source files with model, layer, and device definition |
| torchonn/op      | Basic operators and CUDA-accelerated operators |
| torchonn/layers  | Optical device-implemented layers |
| torchonn/models  | Base ONN model templete |
| torchonn/devices | Optical device parameters and configurations |
| examples/        | ONN model building and training examples |
| examples/configs | YAML-based configuration files|
| examples/core    | ONN model definition and training utility |
| example/train.py | training script|

# More Examples
The `examples/` folder contains more examples to train the ONN
models.

An example optical convolutional neural network `MZI_CLASS_CNN` is defined in `examples/core/models/mzi_cnn.py`.

Training facilities, e.g., optimizer, critetion, lr_scheduler, models are built in `examples/core/builder.py`.
The training and validation logic is defined in `examples/train.py`.
All training hyperparameters are hierarchically defined in the yaml configuration file `examples/configs/mnist/mzi_onn/train.yml` (The final config is the union of all `default.yml` from higher-level directories and this specific `train.yml` ).

By running the following commands,
```python
# train the example MZI-based CNN model with 2 64-channel Conv layers and 1 Linear layer
# training will happend in usv mode to optimize U, Sigma, and V*
# projected gradient descent will be applied to guarantee the orthogonality of U and V*
# the final step will convert unitary matrices into MZI phases and evaluate in the phase mode
cd examples
python3 train.py configs/mnist/mzi_cnn/train.yml # [followed by any command-line arguments that override the values in config file, e.g., --optimizer.lr=0.001]
```

Detailed documentations coming soon.

## Contact
Jiaqi Gu (jqgu@utexas.edu)


# Related Projects using PyTorch-ONN Library
- > **Scalable ONN on-chip learning**: See [L2ight](https://github.com/JeremieMelo/L2ight), NeurIPS 2021.
- > **Memory-efficient ONN architecture**: See [Memory-Efficient-ONN](https://github.com/JeremieMelo/Memory-Efficient-Multi-Level-Generation), ICCV 2021.
- > **SqueezeLight: Scalable ONNs with Multi-Operand Ring Resonators**: See [SqueezeLight](https://github.com/JeremieMelo/SqueezeLight), DATE 2021.
