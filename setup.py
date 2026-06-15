from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import platform
import subprocess

def get_cuda_version():
    """Get CUDA version from nvcc, or None if nvcc is not available."""
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode()
        for line in nvcc_output.split('\n'):
            if 'release' in line.lower():
                return line.split('release')[1].strip().split(',')[0].strip()
    except Exception:
        return None
    return None


class CUDAExtension(Extension):
    """Marker Extension whose single source is a .cu file built by nvcc."""
    def __init__(self, name, sources, *args, **kwargs):
        Extension.__init__(self, name, sources, *args, **kwargs)


class BuildExt(build_ext):
    """Compile each .cu source into a plain shared library (loaded via ctypes).

    The kernels export ``extern "C"`` functions rather than a CPython module,
    so we invoke nvcc directly to emit a shared library at the extension's
    output path. Any failure degrades to a CPU-only install rather than
    aborting the build.
    """

    def build_extension(self, ext):
        source = ext.sources[0]
        output = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(output), exist_ok=True)

        nvcc_flags = ['-O3', '--shared']
        if platform.system() == 'Windows':
            nvcc_flags += ['-Xcompiler', '/MD']
        else:
            nvcc_flags += ['-Xcompiler', '-fPIC']
        if platform.system() == 'Darwin':
            nvcc_flags += ['-Xcompiler', '-stdlib=libc++']

        cmd = ['nvcc'] + nvcc_flags + [source, '-o', output]
        try:
            print(f"Compiling CUDA source {source} -> {output}")
            self.spawn(cmd)
        except Exception as e:
            print(f"WARNING: nvcc failed to build {source}: {e}")
            print("Package will be installed without CUDA acceleration (CPU fallback).")

    def get_ext_filename(self, ext_name):
        """Emit a clean shared-library name (no Python ABI tag) for ctypes."""
        parts = ext_name.split('.')
        if platform.system() == 'Windows':
            suffix = '.dll'
        elif platform.system() == 'Darwin':
            suffix = '.dylib'
        else:
            suffix = '.so'
        return os.path.join(*parts) + suffix


# Only declare CUDA extensions when nvcc is present. Without it the build
# produces a pure-Python (py3-none-any) wheel that runs the CPU fallback.
_cuda_version = get_cuda_version()
if _cuda_version:
    print(f"Building CUDA shared libraries with CUDA {_cuda_version}")
    ext_modules = [
        CUDAExtension(
            "cuda_kernels.autocorrelation._autocorrelation_cuda",
            ["cuda_kernels/autocorrelation/autocorrelation.cu"],
        ),
        CUDAExtension(
            "cuda_kernels.reduction._reduction_cuda",
            ["cuda_kernels/reduction/reduction.cu"],
        ),
    ]
else:
    print("nvcc not found: building CPU-only (pure-Python) package.")
    ext_modules = []

setup(
    name="cuda_kernels",
    version="0.2.0",
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
    ext_modules=ext_modules,
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
