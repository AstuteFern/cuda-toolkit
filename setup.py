from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import platform

class CUDAExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        Extension.__init__(self, name, sources, *args, **kwargs)

class BuildExt(build_ext):
    def build_extensions(self):
        nvcc_flags = ['-O3', '--shared', '--compiler-options', '-fPIC']
        if platform.system() == 'Darwin':
            nvcc_flags.extend(['-Xcompiler', '-stdlib=libc++'])
        
        for ext in self.extensions:
            for i, source in enumerate(ext.sources):
                if source.endswith('.cu'):
                    obj = self.compiler.object_filenames([source])[0]
                    obj = os.path.splitext(obj)[0] + '.o'
                    
                    self.spawn(['nvcc'] + nvcc_flags + ['-o', obj, source])
                    
                    ext.sources[i] = obj
        
        build_ext.build_extensions(self)

setup(
    name="cuda_kernels",
    version="0.1.0",
    author="Sukhman Virk",
    author_email="sukhmanvirk26@gmail.com",
    description="CUDA accelerated correlation and sum reduction functions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AstuteFern/cuda-toolkit",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            "cuda_kernels.autocorrelation.autocorrelation_cuda",
            ["cuda_kernels/autocorrelation/autocorrelation.cu"]
        ),
        CUDAExtension(
            "cuda_kernels.reduction.reduction_cuda",
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
)