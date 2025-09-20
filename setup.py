import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


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
        ort_path, ort_library_path = self._find_onnxruntime_paths()

        import pybind11
        pybind11_cmake_dir = pybind11.get_cmake_dir()
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={python_executable}',
            f'-DORT_PATH={ort_path}',
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
            build_args += ['--', '/m']
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
            import onnxruntime
            onnxruntime_path = Path(onnxruntime.__file__).parent
            print(f"onnxruntime package path: {onnxruntime_path}")

            # Check for capi directory
            capi_path = onnxruntime_path / "capi"
            if capi_path.exists():
                ort_path = str(capi_path)
            else:
                ort_path = str(onnxruntime_path)

            # Find the actual library file
            library_patterns = [
                "libonnxruntime.so.*",
                "libonnxruntime.so",
                "onnxruntime.dll",
                "libonnxruntime.dylib"
            ]

            ort_library_path = None
            for pattern in library_patterns:
                if "*" in pattern:
                    # Handle version-specific patterns
                    import glob
                    matches = glob.glob(os.path.join(ort_path, pattern))
                    if matches:
                        ort_library_path = matches[0]
                        break
                else:
                    lib_path = os.path.join(ort_path, pattern)
                    if os.path.exists(lib_path):
                        ort_library_path = lib_path
                        break

            if not ort_library_path:
                # Fallback: try to find any library file
                for root, dirs, files in os.walk(ort_path):
                    for file in files:
                        if (file.startswith("libonnxruntime") and
                            (file.endswith(".so") or ".so." in file or
                             file.endswith(".dll") or file.endswith(".dylib"))):
                            ort_library_path = os.path.join(root, file)
                            break
                    if ort_library_path:
                        break

            if not ort_library_path:
                raise RuntimeError("Could not find onnxruntime library")

            print(f"Found onnxruntime at: {ort_path}")
            print(f"Found onnxruntime library at: {ort_library_path}")

            return ort_path, ort_library_path

        except ImportError:
            raise RuntimeError("onnxruntime not found. Please install onnxruntime first.")
        except Exception as e:
            raise RuntimeError(f"Error finding onnxruntime: {e}")


# Read pyproject.toml for metadata
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    ext_modules=[CMakeExtension('T2SOnnxCPURuntime', 'runtime')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires=">=3.9",
)