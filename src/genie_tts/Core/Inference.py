import onnxruntime as ort
import numpy as np
from typing import List, Optional
import threading

from ..Audio.ReferenceAudio import ReferenceAudio
from ..Japanese.JapaneseG2P import japanese_to_phones
from ..Utils.Constants import BERT_FEATURE_DIM
from ..ModelManager import GSVModel

class GENIE:
    def __init__(self):
        self.stop_event: threading.Event = threading.Event()

    def tts(
            self,
            text: str,
            prompt_audio: ReferenceAudio,
            gsv_model: GSVModel
    ) -> Optional[np.ndarray]:
        text_seq: np.ndarray = np.array([japanese_to_phones(text)], dtype=np.int64)
        text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
        vocoder: ort.InferenceSession = gsv_model.VITS
        if gsv_model.T2S_CPU_RUNTIME is None:
            semantic_tokens: np.ndarray = self.t2s_backend(
                ref_seq=prompt_audio.phonemes_seq,
                ref_bert=prompt_audio.text_bert,
                text_seq=text_seq,
                text_bert=text_bert,
                ssl_content=prompt_audio.ssl_content,
                encoder=gsv_model.T2S_ENCODER,
                first_stage_decoder=gsv_model.T2S_FIRST_STAGE_DECODER,
                stage_decoder=gsv_model.T2S_STAGE_DECODER,
            )
        else:
            semantic_tokens: np.ndarray = gsv_model.T2S_CPU_RUNTIME.run(
                prompt_audio.phonemes_seq,
                text_seq,
                prompt_audio.text_bert,
                text_bert,
                prompt_audio.ssl_content,
            )
        if self.stop_event.is_set():
            return None

        eos_indices = np.where(semantic_tokens >= 1024)  # 剔除不合法的元素，例如 EOS Token。
        if len(eos_indices[0]) > 0:
            first_eos_index = eos_indices[-1][0]
            semantic_tokens = semantic_tokens[..., :first_eos_index]

        audio_32k = np.expand_dims(prompt_audio.audio_32k, axis=0)  # 增加 Batch_Size 维度
        return vocoder.run(None, {
            "text_seq": text_seq,
            "pred_semantic": semantic_tokens,
            "ref_audio": audio_32k
        })[0]

    def t2s_backend(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
    ) -> Optional[np.ndarray]:
        """在非CPU Backend上运行T2S模型"""
        # Encoder
        x, prompts = encoder.run(
            None,
            {
                "ref_seq": ref_seq,
                "text_seq": text_seq,
                "ref_bert": ref_bert,
                "text_bert": text_bert,
                "ssl_content": ssl_content,
            },
        )
        # First Stage Decoder
        y, y_emb, *present_key_values = first_stage_decoder.run(
            None, {"x": x, "prompts": prompts}
        )
        # Stage Decoder
        input_names: List[str] = [inp.name for inp in stage_decoder.get_inputs()]
        idx: int = 0
        for idx in range(0, 500):
            if self.stop_event.is_set():
                return None
            input_feed = {
                name: data
                for name, data in zip(input_names, [y, y_emb, *present_key_values])
            }
            outputs = stage_decoder.run(None, input_feed)
            y, y_emb, stop_condition_tensor, *present_key_values = outputs

            if stop_condition_tensor:
                break
        y[0, -1] = 0
        return np.expand_dims(y[:, -idx:], axis=0)


tts_client: GENIE = GENIE()
