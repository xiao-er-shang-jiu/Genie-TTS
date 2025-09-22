import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import importlib

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Find Python executable
        python_executable = sys.executable

        # Find onnxruntime paths
        ort_library_path = self._find_onnxruntime_paths()

        pybind11_cmake_dir = pybind11.get_cmake_dir()
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={python_executable}',
            f'-DORT_LIBRARY_PATH={ort_library_path}',
            f'-DCMAKE_BUILD_TYPE=Release',
            f'-Dpybind11_DIR={pybind11_cmake_dir}',
        ]

        build_args = []

        # Platform specific configurations
        if platform.system() == "Windows":
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--config', 'Release', '--', '/m']
        else:
            cmake_args += [f'-DCMAKE_BUILD_TYPE=Release']
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )

        build_temp = self.build_temp
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # Run cmake
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args, cwd=build_temp, env=env
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args, cwd=build_temp
        )

    def _find_onnxruntime_paths(self):
        """Find onnxruntime installation paths"""
        try:
            onnxruntime = importlib.import_module("onnxruntime")
        except ImportError:
            raise ImportError("onnxruntime is required to use this module.")

        ort_dir = Path(onnxruntime.__file__).parent / "capi"
        version = onnxruntime.__version__

        if sys.platform.startswith("linux"):
            libname = "libonnxruntime.so." + version
            return os.path.join(ort_dir, libname)
            
        elif sys.platform == "darwin":
            # macOS
            libname = "libonnxruntime.dylib"
            return os.path.join(ort_dir, libname)
            
        elif sys.platform.startswith("win"):
            # Windows
            libname = "onnxruntime.dll"
            return os.path.join(ort_dir, libname)
            
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

setup(
    ext_modules=[CMakeExtension('T2SOnnxCPURuntime', 'runtime')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires=">=3.9",
)