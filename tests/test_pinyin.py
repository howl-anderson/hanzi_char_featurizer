import numpy as np



class TestPinYin:
    """拼音特征提取器测试"""

    def test_extract_keys(self, pinyin):
        """测试返回正确的键"""
        result = pinyin.extract("明")
        assert "pinyin" in result

    def test_extract_values(self, pinyin):
        """测试返回正确的值类型"""
        result = pinyin.extract("明天")
        assert isinstance(result["pinyin"], list)
        assert len(result["pinyin"]) == 2

    def test_extract_as_numpy(self, pinyin):
        """测试 NumPy 输出"""
        result = pinyin.extract("明天", as_numpy=True)
        assert "pinyin" in result
        assert isinstance(result["pinyin"], np.ndarray)
        assert len(result["pinyin"]) == 2

    def test_vocabulary(self, pinyin):
        """测试词汇表属性"""
        vocab = pinyin.vocabulary
        assert isinstance(vocab, dict)
        assert "pinyin" in vocab
        assert isinstance(vocab["pinyin"], list)

    def test_empty_string(self, pinyin):
        """测试空字符串"""
        result = pinyin.extract("")
        assert result["pinyin"] == []

    def test_special_char(self, pinyin):
        """测试特殊字符"""
        result = pinyin.extract("！😁")
        assert isinstance(result, dict)
        assert len(result["pinyin"]) == 2
