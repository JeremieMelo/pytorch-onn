"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-08 20:49:14
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-08 20:49:14
"""
import os
from distutils import log
from distutils.dir_util import remove_tree

import torch
import torch.cuda
from setuptools import Command, find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension
from torchonn import __version__

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        self.build_base = None
        self.build_lib = None
        self.build_temp = None
        self.build_scripts = None
        self.bdist_base = None
        self.all = None

    def finalize_options(self):
        self.set_undefined_options(
            "build",
            ("build_base", "build_base"),
            ("build_lib", "build_lib"),
            ("build_scripts", "build_scripts"),
            ("build_temp", "build_temp"),
        )
        self.set_undefined_options("bdist", ("bdist_base", "bdist_base"))

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info ./*.so ./torchonn/*.egg-info")
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info ./*.so ./torchonn/*.egg-info")


tokens = str(torch.__version__).split(".")
torch_major_version = "-DTORCH_MAJOR_VERSION=%d" % (int(tokens[0]))
torch_minor_version = "-DTORCH_MINOR_VERSION=%d" % (int(tokens[1]))


def add_prefix(filename):
    return os.path.join("torchonn/op/cuda_extension", filename)


ext_modules = []
if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        "matrix_parametrization_cuda",
        [
            add_prefix("matrix_parametrization/matrix_parametrization_cuda.cpp"),
            add_prefix("matrix_parametrization/matrix_parametrization_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": ["-g", "-fopenmp", "-O2", torch_major_version, torch_minor_version],
            "nvcc": [
                "-O3",
                "-arch=sm_60",
                "-gencode=arch=compute_60,code=sm_60",
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_75,code=compute_75",
                "--use_fast_math",
            ],
        },
    )
    ext_modules.append(extension)

    extension = CUDAExtension(
        "hadamard_cuda",
        [
            add_prefix("hadamard_cuda/hadamard_cuda.cpp"),
            add_prefix("hadamard_cuda/hadamard_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": ["-g", "-fopenmp", "-O2", torch_major_version, torch_minor_version],
            "nvcc": [
                "-O3",
                "-arch=sm_60",
                "-gencode=arch=compute_60,code=sm_60",
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_75,code=compute_75",
                "--use_fast_math",
            ],
        },
    )
    ext_modules.append(extension)

    extension = CUDAExtension(
        "universal_cuda",
        [
            add_prefix("universal_cuda/universal_cuda.cpp"),
            add_prefix("universal_cuda/universal_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": ["-g", "-fopenmp", "-O2", torch_major_version, torch_minor_version],
            "nvcc": [
                "-O3",
                "-arch=sm_60",
                "-gencode=arch=compute_60,code=sm_60",
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_75,code=compute_75",
                "--use_fast_math",
            ],
        },
    )
    ext_modules.append(extension)

setup(
    name="torchonn",
    version=__version__,
    description="Pytorch-centric Optical Neural Network Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JeremieMelo/pytorch-onn",
    author="Jiaqi Gu",
    author_email="jqgu@utexas.edu",
    license="MIT",
    install_requires=[
        "numpy>=1.19.2",
        "torchvision>=0.9.0.dev20210130",
        "tqdm>=4.56.0",
        "setuptools>=52.0.0",
        "torch>=1.8.0",
        "pyutils>=0.0.1",
        "matplotlib>=3.3.4",
        "svglib>=1.1.0",
        "scipy>=1.5.4",
        "scikit-learn>=0.24.1",
        "torchsummary>=1.5.1",
        "pyyaml>=5.1.1",
        "tensorflow-gpu>=2.5.0",
    ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    packages=find_packages(),
    # packages=["models", "layers", "op", "devices"],
    # package_dir={"": "torchonn"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension, "clean": CleanCommand},
)
