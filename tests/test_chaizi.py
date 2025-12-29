import numpy as np



class TestChaiZi:
    """部首拆解特征提取器测试"""

    def test_extract_keys(self, chaizi):
        """测试返回正确的键"""
        result = chaizi.extract("明")
        assert "components" in result

    def test_extract_values(self, chaizi):
        """测试返回正确的值类型"""
        result = chaizi.extract("明天")
        assert isinstance(result["components"], list)
        assert len(result["components"]) == 2

    def test_extract_as_numpy(self, chaizi):
        """测试 NumPy 输出"""
        result = chaizi.extract("明天", as_numpy=True)
        assert "components" in result
        assert isinstance(result["components"], np.ndarray)
        assert len(result["components"]) == 2

    def test_vocabulary(self, chaizi):
        """测试词汇表属性"""
        vocab = chaizi.vocabulary
        assert isinstance(vocab, dict)
        assert "components" in vocab

    def test_empty_string(self, chaizi):
        """测试空字符串"""
        result = chaizi.extract("")
        assert result["components"] == []

    def test_decomposition_result(self, chaizi):
        """测试拆解结果"""
        result = chaizi.extract("闩")
        # 闩 应该被拆解为 门 和 一
        components = result["components"][0]
        assert isinstance(components, (list, tuple))
