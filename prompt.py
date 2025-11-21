import random as rn

class Prompt:
    """
    Convert valence & arousal into a natural language description
    that can be fed to MusicGen.
    """

    def __init__(self):
        # Valence descriptor 리스트
        self.__v_desc = [
            {"start": 0.6, "end": 1.0, "desc": ["Joyful", "Elated", "Bright"]},
            {"start": 0.2, "end": 0.6, "desc": ["Contented", "Warm", "Peaceful"]},
            {"start": -0.2, "end": 0.2, "desc": ["Reflective", "Pensive", "Steady"]},
            {"start": -0.6, "end": -0.2, "desc": ["Somber", "Melancholic", "Heavy"]},
            {"start": -1.0, "end": -0.6, "desc": ["Dissonant", "Ominous", "Aggressive"]},
        ]

        # Arousal descriptor 리스트
        self.__a_desc = [
            {"start": 0.6, "end": 1.0, "desc": ["Driving beat", "Frantic pace", "Energetic percussion"]},
            {"start": 0.2, "end": 0.6, "desc": ["Steady rhythm", "Active pulse", "Flowing beat"]},
            {"start": -0.2, "end": 0.2, "desc": ["Subtle instrumentation", "Calm and smooth tempo", "Minimalist pulse"]},
            {"start": -0.6, "end": -0.2, "desc": ["Sparse texture", "Slow evolving harmony", "Soft and muted rhythm"]},
            {"start": -1.0, "end": -0.6, "desc": ["Almost motionless", "Very slow, drifting", "Distant and static"]},
        ]

    def build_prompt(self, v: float, a: float) -> str:
        """
        Build a full MusicGen prompt from valence & arousal.
        """
        mood = self.__get_desc(self.__v_desc, v)
        motion = self.__get_desc(self.__a_desc, a)
        genre = self.__get_genre(v, a)

        # 최종 프롬프트 (instrumental, no vocals)
        return (
            f"{genre} instrumental soundtrack, "
            f"{mood} mood, {motion}, "
            f"high quality, atmospheric, no vocals."
        )

    def __get_genre(self, v: float, a: float) -> str:
        """
        간단한 사분면 기반 장르 선택
        """
        if v >= 0.3 and a >= 0.3:
            return "Upbeat pop/rock"
        elif v >= 0.3 and a < 0.3:
            return "Warm acoustic or ambient"
        elif v < 0.0 and a >= 0.3:
            return "Dark electronic or cinematic"
        elif v < 0.0 and a < 0.3:
            return "Slow, melancholic piano and strings"
        else:
            return "Neutral lo-fi or ambient"

    def __get_desc(self, desc_list: list, f: float) -> str:
        for desc_range in desc_list:
            if desc_range["start"] <= f < desc_range["end"]:
                rand_index = rn.randint(0, len(desc_range["desc"]) - 1)
                return desc_range["desc"][rand_index]

        # If out of all ranges, clamp to last one
        last = desc_list[-1]
        return last["desc"][rn.randint(0, len(last["desc"]) - 1)]
