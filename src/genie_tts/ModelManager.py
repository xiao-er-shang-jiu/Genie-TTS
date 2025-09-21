import atexit
import gc
from dataclasses import dataclass
import os
import logging
import onnxruntime
from onnxruntime import InferenceSession
from typing import Optional
import numpy as np
# from importlib.resources import files
from huggingface_hub import hf_hub_download

from .Utils.Shared import context
# from .Utils.Constants import PACKAGE_NAME
from .Utils.Utils import LRUCacheDict
import sys, ctypes
from pathlib import Path

logger = logging.getLogger(__name__)

SESS_OPTIONS = onnxruntime.SessionOptions()
SESS_OPTIONS.log_severity_level = 3

# Load ONNX Runtime shared library with global symbol visibility
ort_dir = Path(onnxruntime.__file__).parent / "capi"
version = onnxruntime.__version__
if sys.platform.startswith("linux"):
    libname = "libonnxruntime.so." + version
    mode = 0
    if hasattr(ctypes, "RTLD_NOW"):
        mode |= ctypes.RTLD_NOW
    if hasattr(ctypes, "RTLD_GLOBAL"):
        mode |= ctypes.RTLD_GLOBAL
    lib_path = os.path.join(ort_dir, libname)
    ctypes.CDLL(lib_path, mode=mode)
elif sys.platform == "darwin":
    # macOS
    libname = "libonnxruntime.dylib"
    lib_path = os.path.join(ort_dir, libname)
    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
elif sys.platform.startswith("win"):
    # Windows
    libname = "onnxruntime.dll"
    lib_path = os.path.join(ort_dir, libname)
    ctypes.WinDLL(lib_path)
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")
from T2SOnnxCPURuntime import T2SOnnxCPURuntimeF32


class _GSVModelFile:
    T2S_ENCODER: str = 't2s_encoder_fp32.onnx'
    T2S_FIRST_STAGE_DECODER: str = 't2s_first_stage_decoder_fp32.onnx'
    T2S_STAGE_DECODER: str = 't2s_stage_decoder_fp32.onnx'
    VITS: str = 'vits_fp32.onnx'
    T2S_DECODER_WEIGHT_FP32: str = 't2s_shared_fp32.bin'
    T2S_DECODER_WEIGHT_FP16: str = 't2s_shared_fp16.bin'
    VITS_WEIGHT_FP32: str = 'vits_fp32.bin'
    VITS_WEIGHT_FP16: str = 'vits_fp16.bin'


@dataclass
class GSVModel:
    VITS: InferenceSession
    T2S_ENCODER: Optional[InferenceSession] = None
    T2S_FIRST_STAGE_DECODER: Optional[InferenceSession] = None
    T2S_STAGE_DECODER: Optional[InferenceSession] = None
    T2S_CPU_RUNTIME: Optional[T2SOnnxCPURuntimeF32] = None
    def __post_init__(self):
        # 验证互斥性
        if self.T2S_CPU_RUNTIME is not None and (self.T2S_ENCODER is not None or self.T2S_FIRST_STAGE_DECODER is not None or self.T2S_STAGE_DECODER is not None):
            raise ValueError("T2S_CPU_RUNTIME和其他T2S模型不能同时有值")
        if self.T2S_CPU_RUNTIME is None and (self.T2S_ENCODER is None and self.T2S_FIRST_STAGE_DECODER is None and self.T2S_STAGE_DECODER is None):
            raise ValueError("T2S_CPU_RUNTIME和其他T2S模型必须有一个有值")

def convert_bin_to_fp32(
        fp16_bin_path: str, output_fp32_bin_path: str
) -> None:
    fp16_array = np.fromfile(fp16_bin_path, dtype=np.float16)
    fp32_array = fp16_array.astype(np.float32)
    fp32_array.tofile(output_fp32_bin_path)


def download_model(filename: str, repo_id: str = 'High-Logic/Genie') -> Optional[str]:
    try:
        # package_root = files(PACKAGE_NAME)
        # model_dir = str(package_root / "Data")
        # os.makedirs(model_dir, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            # cache_dir=model_dir,
        )
        return model_path

    except Exception as e:
        logger.error(f"Failed to download model {filename}: {str(e)}", exc_info=True)


def convert_bins_to_fp32(model_dir: str) -> None:
    fp16_fp32_pairs = [
        (_GSVModelFile.T2S_DECODER_WEIGHT_FP16, _GSVModelFile.T2S_DECODER_WEIGHT_FP32),
        (_GSVModelFile.VITS_WEIGHT_FP16, _GSVModelFile.VITS_WEIGHT_FP32),
    ]

    for fp16_name, fp32_name in fp16_fp32_pairs:
        fp16_bin = os.path.normpath(os.path.join(model_dir, fp16_name))
        fp32_bin = os.path.normpath(os.path.join(model_dir, fp32_name))

        if not os.path.exists(fp16_bin):
            raise FileNotFoundError(f"Weight file {fp16_bin} does not exist!")
        if not os.path.exists(fp32_bin):
            convert_bin_to_fp32(fp16_bin, fp32_bin)

    logger.info("Successfully generated temporary FP32 weights to improve inference speed.")


class ModelManager:
    def __init__(self):
        capacity_str = os.getenv('Max_Cached_Character_Models', '3')
        self.character_to_model: dict[str, dict[str, InferenceSession | T2SOnnxCPURuntimeF32]] = LRUCacheDict(
            capacity=int(capacity_str))
        self.character_model_paths: dict[str, str] = {}  # 创建一个持久化字典来存储角色模型路径
        self.providers = ["CPUExecutionProvider"]

        self.cn_hubert: Optional[InferenceSession] = None

    def load_cn_hubert(self) -> bool:
        model_path: Optional[str] = os.getenv("HUBERT_MODEL_PATH")
        if not (model_path and os.path.isfile(model_path)):
            logger.info("Chinese HuBERT model not found locally. Starting download of 'chinese-hubert-base.onnx'...")
            model_path = download_model('chinese-hubert-base.onnx')
            logger.info(f"Chinese HuBERT model download completed. Saved to: {os.path.abspath(model_path)}")
        if not model_path:
            return False
        logger.info(f"Found existing Chinese HuBERT model at: {os.path.abspath(model_path)}")

        try:
            self.cn_hubert = onnxruntime.InferenceSession(model_path,
                                                          providers=self.providers,
                                                          sess_options=SESS_OPTIONS)
            logger.info("Successfully loaded CN_HuBERT model.")
            return True
        except Exception as e:
            logger.error(
                f"Error: Failed to load ONNX model '{model_path}'.\n"
                f"Details: {e}"
            )
        return False

    def get(self, character_name: str) -> Optional[GSVModel]:
        if character_name in self.character_to_model:
            model_map = self.character_to_model[character_name]
            if 'T2SCPURuntime' in model_map and model_map['T2SCPURuntime'] is not None:
                return GSVModel(
                    T2S_CPU_RUNTIME=model_map['T2SCPURuntime'],
                    VITS=model_map[_GSVModelFile.VITS]
                )
            else: 
                return GSVModel(
                    T2S_ENCODER=model_map[_GSVModelFile.T2S_ENCODER],
                    T2S_FIRST_STAGE_DECODER=model_map[_GSVModelFile.T2S_FIRST_STAGE_DECODER],
                    T2S_STAGE_DECODER=model_map[_GSVModelFile.T2S_STAGE_DECODER],
                    VITS=model_map[_GSVModelFile.VITS]
                )
        if character_name in self.character_model_paths:
            model_dir = self.character_model_paths[character_name]
            if self.load_character(character_name, model_dir):
                return self.get(character_name)
            else:
                del self.character_model_paths[character_name]  # 如果重载失败，可以考虑从路径记录中移除，防止反复失败
                return None
        return None

    def has_character(self, character_name: str) -> bool:
        character_name = character_name.lower()
        return character_name in self.character_model_paths

    def load_character(self, character_name: str, model_dir: str) -> bool:
        character_name = character_name.lower()
        if character_name in self.character_to_model:
            logger.info(f"Character '{character_name}' is already in cache; no need to reload.")
            _ = self.character_to_model[character_name]  # 访问一次以更新其在LRU缓存中的位置
            return True

        convert_bins_to_fp32(model_dir)

        model_dict: dict[str, InferenceSession] = {}
        model_filename: list[str] = [_GSVModelFile.T2S_ENCODER,
                                     _GSVModelFile.T2S_FIRST_STAGE_DECODER,
                                     _GSVModelFile.T2S_STAGE_DECODER,
                                     _GSVModelFile.VITS]

        if len(self.providers) == 1 and self.providers[0] == "CPUExecutionProvider":  # Only CPUExecutionProvider
            try:
                model_dict['T2SCPURuntime'] = T2SOnnxCPURuntimeF32(
                    os.path.normpath(os.path.join(model_dir, _GSVModelFile.T2S_ENCODER)),
                    os.path.normpath(os.path.join(model_dir, _GSVModelFile.T2S_FIRST_STAGE_DECODER)),
                    os.path.normpath(os.path.join(model_dir, _GSVModelFile.T2S_STAGE_DECODER))
                )
                model_dict[_GSVModelFile.VITS] = onnxruntime.InferenceSession(
                    os.path.normpath(os.path.join(model_dir, _GSVModelFile.VITS)),
                    providers=self.providers,
                    sess_options=SESS_OPTIONS
                )
                logger.info(f"T2SOnnxCPURuntimeF32 loaded successfully for character '{character_name}'.")
            except Exception as e:
                logger.error(
                    f"Error: Failed to load T2SOnnxCPURuntimeF32 for character '{character_name}'.\n"
                    f"Details: {e}"
                )
                return False
        else:  # Not only CPUExecutionProvider
            for model_file in model_filename:
                model_path: str = os.path.join(model_dir, model_file)
                model_path = os.path.normpath(model_path)
                try:
                    model_dict[model_file] = onnxruntime.InferenceSession(model_path,
                                                                        providers=self.providers,
                                                                        sess_options=SESS_OPTIONS)
                    logger.info(f"Model loaded successfully: {model_path}")
                except Exception as e:
                    logger.error(
                        f"Error: Failed to load ONNX model '{model_path}'.\n"
                        f"Details: {e}"
                    )
                    return False
        

        self.character_to_model[character_name] = model_dict
        self.character_model_paths[character_name] = model_dir

        if not context.current_speaker:
            context.current_speaker = character_name

        return True

    def remove_character(self, character_name: str) -> None:
        character_name = character_name.lower()
        if character_name in self.character_to_model:
            del self.character_to_model[character_name]
            gc.collect()
            logger.info(f"Character {character_name.capitalize()} removed successfully.")

    def clean_cache(self) -> None:
        temp_weights: list[str] = [_GSVModelFile.T2S_DECODER_WEIGHT_FP32, _GSVModelFile.VITS_WEIGHT_FP32]
        deleted_any: bool = False
        try:
            for character, model_dir in self.character_model_paths.items():
                for filename in temp_weights:
                    filepath: str = os.path.join(model_dir, filename)
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        deleted_any = True
            if deleted_any:
                logger.info("All temporary weight files have been successfully deleted.")
        except Exception as e:
            logger.error(f"Failed to delete temporary weight file: {e}")


model_manager: ModelManager = ModelManager()
atexit.register(model_manager.clean_cache)
