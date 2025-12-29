from __future__ import annotations

from functools import cached_property
from typing import Any, Protocol

from numpy.typing import NDArray

from hanzi_char_featurizer.featurizers.four_corner import FourCorner
from hanzi_char_featurizer.featurizers.pinyin_parts import PinYinParts


class FeaturizerProtocol(Protocol):
    """特征器协议"""

    def extract(self, char_seq: str, as_numpy: bool = False) -> dict[str, Any]: ...

    @property
    def vocabulary(self) -> dict[str, Any]: ...


class Featurizer:
    """汉字特征提取器主类"""

    # 默认特征器名称映射
    DEFAULT_FEATURIZER_NAMES: dict[str, str] = {
        "PinYinParts": "pinyin",
        "FourCorner": "four_corner",
    }

    def __init__(self, featurizers: list[FeaturizerProtocol] | None = None) -> None:
        self.featurizers: list[FeaturizerProtocol] = (
            featurizers if featurizers else [PinYinParts(), FourCorner()]
        )

    def _get_featurizer_name(self, featurizer: FeaturizerProtocol) -> str:
        """获取特征器的名称"""
        class_name = featurizer.__class__.__name__
        return self.DEFAULT_FEATURIZER_NAMES.get(class_name, class_name.lower())

    def extract(
        self, char_seq: str, as_numpy: bool = False
    ) -> dict[str, dict[str, Any]] | dict[str, dict[str, NDArray[Any]]]:
        """提取特征

        Args:
            char_seq: 待提取特征的字符序列
            as_numpy: 是否返回 NumPy 数组格式

        Returns:
            嵌套字典，键为特征器名称
        """
        result: dict[str, dict[str, Any]] = {}
        for featurizer in self.featurizers:
            name = self._get_featurizer_name(featurizer)
            result[name] = featurizer.extract(char_seq, as_numpy=as_numpy)
        return result

    @cached_property
    def vocabulary(self) -> dict[str, dict[str, Any]]:
        """词汇表"""
        result: dict[str, dict[str, Any]] = {}
        for featurizer in self.featurizers:
            name = self._get_featurizer_name(featurizer)
            result[name] = featurizer.vocabulary
        return result


if __name__ == "__main__":
    featurizer = Featurizer()

    print("extract():", featurizer.extract("明天"))
    print("extract(as_numpy=True):", featurizer.extract("明天", as_numpy=True))
    print("vocabulary:", featurizer.vocabulary)
