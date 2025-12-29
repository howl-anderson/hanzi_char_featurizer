import numpy as np



class TestFourCorner:
    """四角编码特征提取器测试"""

    def test_extract_keys(self, four_corner):
        """测试返回正确的键"""
        result = four_corner.extract("明")
        expected_keys = {"upper_left", "upper_right", "lower_left", "lower_right", "extra"}
        assert set(result.keys()) == expected_keys

    def test_extract_values(self, four_corner):
        """测试返回正确的值类型"""
        result = four_corner.extract("明天")
        for key, value in result.items():
            assert isinstance(value, list)
            assert len(value) == 2  # 两个字符

    def test_extract_as_numpy(self, four_corner):
        """测试 NumPy 输出"""
        result = four_corner.extract("明天", as_numpy=True)
        for key, value in result.items():
            assert isinstance(value, np.ndarray)
            assert len(value) == 2

    def test_vocabulary(self, four_corner):
        """测试词汇表属性"""
        vocab = four_corner.vocabulary
        assert isinstance(vocab, dict)
        assert "upper_left" in vocab
        assert len(vocab["upper_left"]) == 10  # 0-9

    def test_empty_string(self, four_corner):
        """测试空字符串"""
        result = four_corner.extract("")
        for key, value in result.items():
            assert value == []

    def test_non_chinese_char(self, four_corner):
        """测试非中文字符"""
        # 非中文字符应该也能处理（可能返回默认值）
        result = four_corner.extract("A")
        assert isinstance(result, dict)
