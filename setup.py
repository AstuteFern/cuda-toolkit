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
            print("WARNING: CUDA is not available. Package will be installed without CUDA acceleration.")
            print("CPU fallback implementations will be used.")
            print("To enable CUDA acceleration, install CUDA toolkit and reinstall this package.")
            # Remove CUDA extensions if CUDA is not available
            self.extensions = []
            return

        print(f"Building with CUDA version {cuda_version}")
        
        # Check if PyTorch CUDA is available (optional for better integration)
        try:
            import torch
            print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
        except ImportError:
            print("PyTorch not found - CUDA extensions will still be built")

        # Set up NVCC flags
        nvcc_flags = ['-O3', '--shared']
        if platform.system() == 'Windows':
            nvcc_flags.extend(['--compiler-options', '/MD'])
        else:
            nvcc_flags.extend(['--compiler-options', '-fPIC'])
            
        if platform.system() == 'Darwin':
            nvcc_flags.extend(['-Xcompiler', '-stdlib=libc++'])
        
        # Compile CUDA sources
        for ext in self.extensions:
            for i, source in enumerate(ext.sources):
                if source.endswith('.cu'):
                    if platform.system() == 'Windows':
                        obj = os.path.splitext(source)[0] + '.obj'
                    else:
                        obj = os.path.splitext(source)[0] + '.o'
                    
                    try:
                        cmd = ['nvcc'] + nvcc_flags + ['-c', source, '-o', obj]
                        print(f"Compiling {source}...")
                        self.spawn(cmd)
                        ext.sources[i] = obj
                    except Exception as e:
                        print(f"WARNING: Failed to compile CUDA source {source}: {str(e)}")
                        print("Package will be installed without CUDA acceleration.")
                        self.extensions = []
                        return
        
        try:
            build_ext.build_extensions(self)
        except Exception as e:
            print(f"WARNING: Failed to build CUDA extensions: {str(e)}")
            print("Package will be installed without CUDA acceleration.")
            self.extensions = []

    def get_ext_filename(self, ext_name):
        """Get the filename for the extension module."""
        if platform.system() == 'Windows':
            return ext_name + '.pyd'
        return super().get_ext_filename(ext_name)

setup(
    name="cuda_kernels",
    version="0.1.1",
    author="Sukhman Virk, Shiv Mehta",
    author_email="sukhmanvirk26@gmail.com",
    description="CUDA accelerated correlation and sum reduction functions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AstuteFern/cuda-toolkit",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'cuda_kernels': ['*/*.cu'],
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
    ],
    extras_require={
        "cuda": ["torch>=1.7.0"],
    },
)
