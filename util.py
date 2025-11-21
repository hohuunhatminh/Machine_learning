import torch
import numpy as np

TEXT_W = 0.3   # how much we trust text vs. face


def map_text(text_data: dict) -> dict:
    """
    Map text emotion probabilities to valence-arousal.
    예상 라벨: angry, fear, happy, neutral, sad, surprise
    """
    va = {"valence": 0.0, "arousal": 0.0}

    emo = text_data.get("angry", 0.0)
    va["valence"] += emo * -0.6
    va["arousal"] += emo * 0.7

    emo = text_data.get("fear", 0.0)
    va["valence"] += emo * -0.7
    va["arousal"] += emo * 0.6

    emo = text_data.get("happy", 0.0)
    va["valence"] += emo * 0.8
    va["arousal"] += emo * 0.4

    emo = text_data.get("sad", 0.0)
    va["valence"] += emo * -0.8
    va["arousal"] += emo * -0.3

    emo = text_data.get("surprise", 0.0)
    va["valence"] += emo * 0.2
    va["arousal"] += emo * 0.7

    emo = text_data.get("neutral", 0.0)
    # neutral ~ center
    va["valence"] += emo * 0.0
    va["arousal"] += emo * 0.0

    va["valence"] = float(np.clip(va["valence"], -1.0, 1.0))
    va["arousal"] = float(np.clip(va["arousal"], -1.0, 1.0))
    return va


def map_face(face_data: dict) -> dict:
    """
    DeepFace emotion 결과를 valence-arousal에 매핑.
    DeepFace labels: angry, disgust, fear, happy, sad, surprise, neutral
    """
    va = {"valence": 0.0, "arousal": 0.0}

    emo = face_data.get("angry", 0.0)
    va["valence"] += emo * -0.7
    va["arousal"] += emo * 0.6

    emo = face_data.get("disgust", 0.0)
    va["valence"] += emo * -0.7
    va["arousal"] += emo * 0.3

    emo = face_data.get("fear", 0.0)
    va["valence"] += emo * -0.7
    va["arousal"] += emo * 0.7

    emo = face_data.get("happy", 0.0)
    va["valence"] += emo * 0.8
    va["arousal"] += emo * 0.5

    emo = face_data.get("sad", 0.0)
    va["valence"] += emo * -0.8
    va["arousal"] += emo * -0.4

    emo = face_data.get("surprise", 0.0)
    va["valence"] += emo * 0.1
    va["arousal"] += emo * 0.8

    emo = face_data.get("neutral", 0.0)
    va["valence"] += emo * 0.0
    va["arousal"] += emo * 0.0

    va["valence"] = float(np.clip(va["valence"], -1.0, 1.0))
    va["arousal"] = float(np.clip(va["arousal"], -1.0, 1.0))
    return va


def fuse(text_va: dict | None, face_va: dict | None, text_w: float = TEXT_W) -> dict:
    """
    Fuse valence/arousal from text and face.
    text_w: weight for text; (1 - text_w) used for face.
    """
    if text_va is None and face_va is None:
        raise ValueError("At least one of text_va or face_va must be provided.")

    if text_va is None:
        return dict(face_va)
    if face_va is None:
        return dict(text_va)

    text_w = float(np.clip(text_w, 0.0, 1.0))
    face_w = 1.0 - text_w

    valence = text_w * text_va["valence"] + face_w * face_va["valence"]
    arousal = text_w * text_va["arousal"] + face_w * face_va["arousal"]

    return {
        "valence": float(np.clip(valence, -1.0, 1.0)),
        "arousal": float(np.clip(arousal, -1.0, 1.0)),
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vec_softmax(t: torch.Tensor) -> list[float]:
    vec = [float(el) for el in t[0]]
    e_vec = np.exp(vec - np.max(vec))
    return list(e_vec / e_vec.sum())
