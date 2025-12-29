from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pypinyin import pinyin, Style
from pypinyin.style._constants import _INITIALS


raw_FINALS = "i,u,v,a,ia,ua,o,uo,e,ie,ve,ai,uai,ei,uei,ao,iao,ou,iou,an,ian,uan,van,en,in,uen,vn,ang,iang,uang,eng,ing,ueng,ong,iong".split(
    ","
)

# ordered by length, so FINALS can work with endswith correctly
FINALS: list[str] = sorted(raw_FINALS, key=lambda x: len(x), reverse=True)


class PinYinParts:
    # 特征键名
    FEATURE_KEYS: list[str] = ["initial", "final", "tone"]

    common_params: dict[str, Any] = {
        "heteronym": True,
        "errors": lambda x: [i for i in x],  # return char literally
    }

    padding: tuple[str, str, str] = ("-", "-", "0")

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params if params else {"pad_len": 1}

    def _extract_raw(self, char_seq: str) -> list[list[tuple[str, str, str]]]:
        char_pinyin_seq = pinyin(char_seq, style=Style.TONE3, **self.common_params)

        initial_result = []
        final_result = []
        tone_result = []

        for char_pinyin in char_pinyin_seq:
            char_initial = []
            char_final = []
            char_tone = []
            for candidate_pinyin in char_pinyin:
                if candidate_pinyin == "-":
                    char_initial.append("-")
                    char_final.append("-")
                    char_tone.append("0")

                    continue

                initial = "-"
                # _INITIALS has a special order can work with startswith() correctly
                for i in _INITIALS:
                    if candidate_pinyin.startswith(i):
                        initial = i
                        break
                char_initial.append(initial)

                # tone part
                tone = candidate_pinyin[-1]
                char_tone.append(tone)

                # final part
                candidate_pinyin_without_tone = candidate_pinyin[:-1]
                final = "-"
                # FINALS has a special order can work with endswith() correctly
                for i in FINALS:
                    if candidate_pinyin_without_tone.endswith(i):
                        final = i
                        break
                char_final.append(final)

            initial_result.append(char_initial)
            final_result.append(char_final)
            tone_result.append(char_tone)

        # zip parts together
        result = []
        for char_index in range(len(initial_result)):
            item = list(
                zip(initial_result[char_index], final_result[char_index], tone_result[char_index])
            )
            result.append(item)

        # padding parts
        padded_result = []
        for item in result:
            padding_length = max((self.params["pad_len"] - len(item), 0))
            padding_part = [self.padding] * padding_length
            after_padding = item + padding_part

            # make sure no longer than this
            fine_item = after_padding[: self.params["pad_len"]]

            padded_result.append(fine_item)

        return padded_result

    def extract(
        self, char_seq: str, as_numpy: bool = False
    ) -> dict[str, list[list[str]]] | dict[str, NDArray[np.object_]]:
        """提取特征

        Args:
            char_seq: 待提取特征的字符序列
            as_numpy: 是否返回 NumPy 数组格式

        Returns:
            特征字典
        """
        padded_result = self._extract_raw(char_seq)

        feature_data: dict[str, list[list[str]]] = {"initial": [], "final": [], "tone": []}

        for char_feature in padded_result:
            initial_feature: list[str] = []
            final_feature: list[str] = []
            tone_feature: list[str] = []

            for candidate_feature in char_feature:
                initial_feature.append(candidate_feature[0])
                final_feature.append(candidate_feature[1])
                tone_feature.append(candidate_feature[2])

            feature_data["initial"].append(initial_feature)
            feature_data["final"].append(final_feature)
            feature_data["tone"].append(tone_feature)

        if as_numpy:
            return {key: np.array(values, dtype=object) for key, values in feature_data.items()}
        return feature_data

    @cached_property
    def vocabulary(self) -> dict[str, list[str]]:
        """词汇表"""
        return {"initial": list(_INITIALS), "final": FINALS, "tone": [str(i) for i in range(5)]}


if __name__ == "__main__":
    obj = PinYinParts()
    print("extract():", obj.extract("明天"))
    print("extract(as_numpy=True):", obj.extract("明天", as_numpy=True))
    print("vocabulary:", obj.vocabulary)
