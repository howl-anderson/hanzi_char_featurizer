from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pypinyin import pinyin, Style
from pypinyin.style.bopomofo import BOPOMOFO_TABLE


class PinYin:
    # 特征键名
    FEATURE_KEYS: list[str] = ["pinyin"]

    default_params: dict[str, Any] = {"style": Style.BOPOMOFO, "errors": lambda x: [i for i in x]}

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params if params else self.default_params

    def extract(
        self, char_seq: str, as_numpy: bool = False
    ) -> dict[str, list[str]] | dict[str, NDArray[np.str_]]:
        """提取特征

        Args:
            char_seq: 待提取特征的字符序列
            as_numpy: 是否返回 NumPy 数组格式

        Returns:
            特征字典
        """
        raw_result = pinyin(char_seq, **self.params)
        result = [i[0] for i in raw_result]

        if as_numpy:
            return {"pinyin": np.array(result)}
        return {"pinyin": result}

    @cached_property
    def vocabulary(self) -> dict[str, list[str]]:
        """词汇表"""
        return {"pinyin": list(BOPOMOFO_TABLE.values())}


if __name__ == "__main__":
    obj = PinYin()
    print("extract():", obj.extract("明天"))
    print("extract(as_numpy=True):", obj.extract("明天", as_numpy=True))
    print("vocabulary:", obj.vocabulary)
