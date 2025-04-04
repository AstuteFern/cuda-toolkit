from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import platform
import sys
import subprocess

def get_cuda_version():
    """Get CUDA version from nvcc."""
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode()
        for line in nvcc_output.split('\n'):
            if 'release' in line.lower():
                return line.split('release')[1].strip().split(',')[0].strip()
    except:
        return None
    return None

class CUDAExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        Extension.__init__(self, name, sources, *args, **kwargs)

class BuildExt(build_ext):
    def build_extensions(self):
        # Check CUDA availability
        cuda_version = get_cuda_version()
        if not cuda_version:
            print("ERROR: CUDA is not available. Please install CUDA toolkit and ensure nvcc is in your PATH.")
            print("You can download CUDA from: https://developer.nvidia.com/cuda-downloads")
            sys.exit(1)

        print(f"Building with CUDA version {cuda_version}")
        
        # Check if PyTorch CUDA is available
        try:
            import torch
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"System CUDA version: {get_cuda_version()}")  # From your setup.py
            if not torch.cuda.is_available():
                print("ERROR: PyTorch CUDA is not available. Please install PyTorch with CUDA support.")
                print("You can install it using: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                sys.exit(1)
        except ImportError:
            print("ERROR: PyTorch is not installed. Please install PyTorch with CUDA support.")
            print("You can install it using: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            sys.exit(1)

        nvcc_flags = ['-O3', '--shared', '--compiler-options', '-fPIC']
        if platform.system() == 'Darwin':
            nvcc_flags.extend(['-Xcompiler', '-stdlib=libc++'])
        
        for ext in self.extensions:
            for i, source in enumerate(ext.sources):
                if source.endswith('.cu'):
                    obj = self.compiler.object_filenames([source])[0]
                    obj = os.path.splitext(obj)[0] + '.o'
                    
                    try:
                        self.spawn(['nvcc'] + nvcc_flags + ['-o', obj, source])
                        ext.sources[i] = obj
                    except Exception as e:
                        print(f"ERROR: Failed to compile CUDA source {source}: {str(e)}")
                        sys.exit(1)
        
        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        """Get the filename for the extension module."""
        if platform.system() == 'Windows':
            return ext_name + '.pyd'
        return super().get_ext_filename(ext_name)

setup(
    name="cuda_kernels",
    version="0.1.0",
    author="Sukhman Virk, Shiv Mehta",
    author_email="sukhmanvirk26@gmail.com",
    description="CUDA accelerated correlation and sum reduction functions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AstuteFern/cuda-toolkit",
    packages=find_packages(include=['cuda_kernels', 'cuda_kernels.*']),
    include_package_data=True,
    package_data={
        'cuda_kernels.autocorrelation': ['*.cu'],
        'cuda_kernels.reduction': ['*.cu']
    },
    ext_modules=[
        CUDAExtension(
            "cuda_kernels.autocorrelation._autocorrelation_cuda",
            ["cuda_kernels/autocorrelation/autocorrelation.cu"]
        ),
        CUDAExtension(
            "cuda_kernels.reduction._reduction_cuda",
            ["cuda_kernels/reduction/reduction.cu"]
        )
    ],
    cmdclass={'build_ext': BuildExt},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.16.0",
        "torch>=1.7.0",
    ],
)
