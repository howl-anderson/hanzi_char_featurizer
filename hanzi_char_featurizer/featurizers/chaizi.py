from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hanzi_chaizi import HanziChaizi


class ChaiZi:
    # 特征键名
    FEATURE_KEYS: list[str] = ["components"]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params if params else {}
        self.hc = HanziChaizi(**self.params)

    def extract(
        self, char_seq: str, as_numpy: bool = False
    ) -> dict[str, list[tuple[str, ...] | None]] | dict[str, NDArray[np.object_]]:
        """提取特征

        Args:
            char_seq: 待提取特征的字符序列
            as_numpy: 是否返回 NumPy 数组格式

        Returns:
            特征字典
        """
        result = [self.hc.query(i) for i in char_seq]

        if as_numpy:
            return {"components": np.array(result, dtype=object)}
        return {"components": result}

    @cached_property
    def vocabulary(self) -> dict[str, None]:
        """词汇表（部首拆解的词汇表较大，按需获取）"""
        return {"components": None}


if __name__ == "__main__":
    obj = ChaiZi()
    print("extract():", obj.extract("明天"))
    print("extract(as_numpy=True):", obj.extract("明天", as_numpy=True))
    print("vocabulary:", obj.vocabulary)
