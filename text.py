
import torch
from util import vec_softmax
from transformers import (
    AutoModelForSequenceClassification as AutoCls,
    AutoTokenizer as AutoTok,
)

REF = "Jinuuuu/KoELECTRA_fine_tunning_emotion"


class TextModel:
    def __init__(self) -> None:
        try:
            self.__model = AutoCls.from_pretrained(REF)
            self.__tokenizer = AutoTok.from_pretrained(REF)
            self.__labels = list(self.__model.config.id2label.values())
            self.__model.eval()
        except Exception as e:
            print(f"TextModel Error: {e}")
            self.__model = None
            self.__tokenizer = None
            self.__labels = []

    def predict(self, text: str) -> dict:
        """
        :param text: Korean diary sentence(s)
        :return: {label: probability}
        """
        if not self.__model or not self.__tokenizer:
            print("TextModel Error: model/tokenizer not initialized.")
            return {}

        if not text or not text.strip():
            return {}

        try:
            inputs = self.__tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )

            with torch.no_grad():
                output = self.__model(**inputs)
                prob = vec_softmax(output.logits)

            emo = {}
            for i, p in enumerate(prob):
                label = self.__labels[i]
                emo[label] = float(p)

            return emo

        except Exception as e:
            print(f"TextModel Error: {e}")
            return {}
