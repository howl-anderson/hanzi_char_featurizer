import numpy as np



class TestPinYinParts:
    """拼音部分特征提取器测试"""

    def test_extract_keys(self, pinyin_parts):
        """测试返回正确的键"""
        result = pinyin_parts.extract("明")
        expected_keys = {"initial", "final", "tone"}
        assert set(result.keys()) == expected_keys

    def test_extract_values(self, pinyin_parts):
        """测试返回正确的值"""
        result = pinyin_parts.extract("明天")
        assert result["initial"] == [["m"], ["t"]]
        assert result["final"] == [["ing"], ["ian"]]
        assert result["tone"] == [["2"], ["1"]]

    def test_extract_as_numpy(self, pinyin_parts):
        """测试 NumPy 输出"""
        result = pinyin_parts.extract("明天", as_numpy=True)
        for key, value in result.items():
            assert isinstance(value, np.ndarray)
            assert len(value) == 2

    def test_vocabulary(self, pinyin_parts):
        """测试词汇表属性"""
        vocab = pinyin_parts.vocabulary
        assert isinstance(vocab, dict)
        assert "initial" in vocab
        assert "final" in vocab
        assert "tone" in vocab
        assert "0" in vocab["tone"]  # 轻声
        assert "1" in vocab["tone"]  # 一声

    def test_empty_string(self, pinyin_parts):
        """测试空字符串"""
        result = pinyin_parts.extract("")
        for key, value in result.items():
            assert value == []

    def test_special_char(self, pinyin_parts):
        """测试特殊字符（标点、emoji）"""
        result = pinyin_parts.extract("！")
        assert isinstance(result, dict)

    def test_polyphonic_char(self, pinyin_parts):
        """测试多音字"""
        result = pinyin_parts.extract("行")
        # 行是多音字，heteronym=True 应该返回多个读音
        assert "initial" in result
        # 检查是否有多个读音候选
        assert len(result["initial"][0]) >= 1

    def test_tone_values(self, pinyin_parts):
        """测试声调值范围"""
        # 测试四个声调
        result = pinyin_parts.extract("妈麻马骂")
        tones = [t[0] for t in result["tone"]]
        assert "1" in tones
        assert "2" in tones
        assert "3" in tones
        assert "4" in tones
