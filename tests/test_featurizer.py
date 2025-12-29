import numpy as np

from hanzi_char_featurizer import Featurizer
from hanzi_char_featurizer.featurizers.four_corner import FourCorner
from hanzi_char_featurizer.featurizers.pinyin_parts import PinYinParts


class TestFeaturizer:
    """主特征提取器测试"""

    def test_default_featurizers(self, featurizer):
        """测试默认特征器"""
        assert len(featurizer.featurizers) == 2
        assert isinstance(featurizer.featurizers[0], PinYinParts)
        assert isinstance(featurizer.featurizers[1], FourCorner)

    def test_custom_featurizers(self):
        """测试自定义特征器"""
        custom = Featurizer(featurizers=[FourCorner()])
        assert len(custom.featurizers) == 1

    def test_extract_structure(self, featurizer):
        """测试 extract 返回结构"""
        result = featurizer.extract("明天")
        assert isinstance(result, dict)
        assert "pinyin" in result
        assert "four_corner" in result

    def test_extract_pinyin_content(self, featurizer):
        """测试拼音部分内容"""
        result = featurizer.extract("明天")
        pinyin = result["pinyin"]
        assert "initial" in pinyin
        assert "final" in pinyin
        assert "tone" in pinyin

    def test_extract_four_corner_content(self, featurizer):
        """测试四角编码部分内容"""
        result = featurizer.extract("明天")
        fc = result["four_corner"]
        assert "upper_left" in fc
        assert "upper_right" in fc
        assert "lower_left" in fc
        assert "lower_right" in fc
        assert "extra" in fc

    def test_extract_as_numpy(self, featurizer):
        """测试 NumPy 输出"""
        result = featurizer.extract("明天", as_numpy=True)
        assert isinstance(result, dict)
        # 验证嵌套的值是 NumPy 数组
        for name, features in result.items():
            for key, value in features.items():
                assert isinstance(value, np.ndarray)

    def test_vocabulary(self, featurizer):
        """测试词汇表属性"""
        vocab = featurizer.vocabulary
        assert isinstance(vocab, dict)
        assert "pinyin" in vocab
        assert "four_corner" in vocab

    def test_empty_string(self, featurizer):
        """测试空字符串"""
        result = featurizer.extract("")
        assert isinstance(result, dict)

    def test_single_char(self, featurizer):
        """测试单个字符"""
        result = featurizer.extract("明")
        assert result["pinyin"]["initial"] == [["m"]]
        assert result["pinyin"]["final"] == [["ing"]]
        assert result["pinyin"]["tone"] == [["2"]]


class TestEdgeCases:
    """边界情况测试"""

    def test_special_characters(self, featurizer):
        """测试特殊字符"""
        result = featurizer.extract("！@#")
        assert isinstance(result, dict)

    def test_mixed_content(self, featurizer):
        """测试中英混合"""
        result = featurizer.extract("Hello你好")
        assert isinstance(result, dict)

    def test_emoji(self, featurizer):
        """测试 emoji"""
        result = featurizer.extract("😁")
        assert isinstance(result, dict)

    def test_long_text(self, featurizer):
        """测试长文本"""
        text = "这是一段比较长的测试文本用于验证系统的稳定性"
        result = featurizer.extract(text)
        assert len(result["pinyin"]["initial"]) == len(text)
