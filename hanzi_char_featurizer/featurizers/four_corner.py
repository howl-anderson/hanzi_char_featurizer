from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from four_corner_method import FourCornerMethod


class FourCorner:
    # 特征键名
    FEATURE_KEYS: list[str] = ["upper_left", "upper_right", "lower_left", "lower_right", "extra"]

    # 非汉字字符的默认编码
    DEFAULT_CODE: tuple[str, ...] = ("0", "0", "0", "0", "0")

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params if params else {}
        self.fcm = FourCornerMethod(**self.params)

    def _extract_raw(self, char_seq: str) -> list[list[str]]:
        result: list[list[str]] = []
        for char in char_seq:
            query_result = self.fcm.query(char)
            if query_result is None:
                result.append(list(self.DEFAULT_CODE))
            else:
                result.append([i for i in query_result])
        return result

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
        raw_result = self._extract_raw(char_seq)
        transposed = list(zip(*raw_result)) if raw_result else [[] for _ in self.FEATURE_KEYS]
        result = {key: list(values) for key, values in zip(self.FEATURE_KEYS, transposed)}

        if as_numpy:
            return {key: np.array(values) for key, values in result.items()}
        return result

    @cached_property
    def vocabulary(self) -> dict[str, list[str]]:
        """词汇表"""

        def range_str(x: int) -> list[str]:
            return [str(i) for i in range(x)]

        return {
            "upper_left": range_str(10),
            "upper_right": range_str(10),
            "lower_left": range_str(10),
            "lower_right": range_str(10),
            "extra": range_str(10),
        }


if __name__ == "__main__":
    obj = FourCorner()
    print("extract():", obj.extract("明天"))
    print("extract(as_numpy=True):", obj.extract("明天", as_numpy=True))
    print("vocabulary:", obj.vocabulary)
