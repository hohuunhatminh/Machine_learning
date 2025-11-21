import os
from pathlib import Path
from typing import Optional

from transformers import pipeline
import scipy.io.wavfile as wavfile


class MusicModel:
    """
    Simple wrapper around MusicGen (facebook/musicgen-small).

    - 입력: 자연어 프롬프트 (예: "warm, peaceful piano and strings...")
    - 출력: 프로젝트 폴더 안의 EmoMaestro.wav 파일 경로
    """

    def __init__(self,
                 model_name: str = "facebook/musicgen-small",
                 out_name: str = "EmoMaestro.wav") -> None:
        self._model_name = model_name
        self._out_name = out_name
        self._pipe = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            # HuggingFace transformers text-to-audio pipeline
            self._pipe = pipeline("text-to-audio", model=self._model_name)
        except Exception as e:
            print(f"MusicModel Error while loading model: {e}")
            self._pipe = None

    def gen(self, prompt: str, dur_sec: int = 30) -> Optional[str]:
        """
        Generate music from a text prompt.

        :param prompt: text prompt describing mood/genre/instrument
        :param dur_sec: target duration in seconds (approximate)
        :return: absolute path to generated wav file, or None on error
        """
        if not self._pipe:
            print("MusicModel Error: pipeline is not initialized.")
            return None

        if not prompt or not prompt.strip():
            print("MusicModel Error: empty prompt.")
            return None

        # Roughly convert seconds → token length
        # (MusicGen length is actually token-based, so this is only an approximation)
        tokens_per_second = 50
        max_new_tokens = max(64, int(tokens_per_second * dur_sec))

        try:
            output = self._pipe(
                prompt,
                forward_params={
                    "do_sample": True,
                    "max_new_tokens": max_new_tokens,
                },
            )

            # pipeline may return list or single dict
            if isinstance(output, list):
                output = output[0]

            audio = output["audio"]
            sr = output["sampling_rate"]

            # Save into project directory with fixed name
            out_path = Path(__file__).with_name(self._out_name)
            wavfile.write(out_path, rate=sr, data=audio)

            return str(out_path.resolve())

        except Exception as e:
            print(f"MusicModel Error during generation: {e}")
            return None
