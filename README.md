# 汉字字符特征提取器（featurizer）

在深度学习中，很多场合需要提取汉字的特征（发音特征、字形特征）。本项目提供了一个通用的字符特征提取框架，并内建了 `拼音`、`字形`（四角编码） 和 `部首拆解` 的特征。

## 使用
```python
from hanzi_char_featurizer import featurize

result = featurize('明天')
print(result)
```

输出
```text
[('míng', '67020', ['日', '月']), ('tiān', '10804', ['一', '大'])]
```
