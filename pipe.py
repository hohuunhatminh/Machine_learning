from util import map_text, map_face, fuse, TEXT_W
from text import TextModel
from face import FaceModel
from music import MusicModel
from prompt import Prompt
# from playsound import playsound   # 필요하면 사용

def pipe(text: str, img_path: str, text_w: float = TEXT_W) -> None:
    """
    Main pipeline:
    1) Get emotion from text and/or face
    2) Map to valence-arousal space
    3) Create natural language prompt
    4) Generate music with MusicGen
    """
    try:
        text = (text or "").strip()
        img_path = (img_path or "").strip()

        music_model = MusicModel()
        prompt_model = Prompt()

        text_va = None
        face_va = None

        print(f"\nText Diary: {text}")
        print(f"Portrait: {img_path}\n")

        # 1. Text → emotion → (v, a)
        if text:
            text_model = TextModel()
            text_emo = text_model.predict(text)
            print(f"Text emotion probabilities: {text_emo}")
            text_va = map_text(text_emo)
            print(f"Text valence/arousal: {text_va}")

        # 2. Face → emotion → (v, a)
        if img_path:
            face_model = FaceModel()
            face_emo = face_model.predict(img_path)
            print(f"Face emotion probabilities: {face_emo}")
            face_va = map_face(face_emo)
            print(f"Face valence/arousal: {face_va}")

        if text_va is None and face_va is None:
            raise Exception("No valid emotion source (both text and face are empty).")

        # 3. Fuse 두 소스
        va = fuse(text_va, face_va, text_w=text_w)
        print(f"Fused valence/arousal: {va}")

        v = va["valence"]
        a = va["arousal"]

        # 4. Valence/Arousal → 자연어 프롬프트
        p = prompt_model.build_prompt(v, a)
        print(f"\nPrompt: {p}\n")

        # 5. MusicGen으로 음악 생성
        audio_path = music_model.gen(p)

        print(f"\nAudio path: {audio_path}\n")

        # if audio_path:
        #     playsound(audio_path)

    except Exception as e:
        print(f"Pipeline Error: {e}")
