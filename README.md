# Hanzi Char Featurizer / 汉字字符特征提取器

[![PyPI version](https://badge.fury.io/py/hanzi-char-featurizer.svg)](https://badge.fury.io/py/hanzi-char-featurizer)
[![Python](https://img.shields.io/pypi/pyversions/hanzi-char-featurizer.svg)](https://pypi.org/project/hanzi-char-featurizer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Extract multi-dimensional features from Chinese characters for deep learning: phonetic features, glyph features, and structural features.

为深度学习应用提取汉字的多维特征：发音特征、字形特征、结构特征。

## Feature Extractors / 特征提取器

| Extractor / 特征器 | Description / 说明 | Example / 示例 |
|-------------------|-------------------|----------------|
| **PinYinParts** | Pinyin decomposition (initial, final, tone) / 拼音分解（声母、韵母、声调） | `明` → `{m, ing, 2}` |
| **FourCorner** | Four-corner encoding / 四角号码编码 | `明` → `{6, 7, 0, 2, 0}` |
| **ChaiZi** | Radical decomposition / 部首拆解 | `明` → `(日, 月)` |

## Installation / 安装

```bash
pip install hanzi_char_featurizer
```

## Quick Start / 快速开始

```python
from hanzi_char_featurizer import Featurizer

featurizer = Featurizer()

# Extract features / 提取特征
result = featurizer.extract('明天')
print(result)
```

Output / 输出：
```python
{
    'pinyin': {
        'initial': [['m'], ['t']],
        'final': [['ing'], ['ian']],
        'tone': [['2'], ['1']]
    },
    'four_corner': {
        'upper_left': ['6', '1'],
        'upper_right': ['7', '0'],
        'lower_left': ['0', '8'],
        'lower_right': ['2', '0'],
        'extra': ['0', '4']
    }
}
```

## API

```python
# Extract features (returns dict) / 提取特征（返回 dict）
result = featurizer.extract('明天')

# Extract features (returns NumPy arrays) / 提取特征（返回 NumPy 数组）
result = featurizer.extract('明天', as_numpy=True)

# Get vocabulary / 获取词汇表
vocab = featurizer.vocabulary
```

## Using Individual Extractors / 单独使用特征器

```python
from hanzi_char_featurizer.featurizers.four_corner import FourCorner
from hanzi_char_featurizer.featurizers.pinyin_parts import PinYinParts
from hanzi_char_featurizer.featurizers.chaizi import ChaiZi

fc = FourCorner()
fc.extract('明')  # {'upper_left': ['6'], 'upper_right': ['7'], ...}

pp = PinYinParts()
pp.extract('明')  # {'initial': [['m']], 'final': [['ing']], 'tone': [['2']]}

cz = ChaiZi()
cz.extract('明')  # {'components': [('日', '月')]}
```

## Custom Extractor Combination / 自定义特征器组合

```python
from hanzi_char_featurizer import Featurizer
from hanzi_char_featurizer.featurizers.four_corner import FourCorner

# Use only four-corner encoding / 只使用四角编码
featurizer = Featurizer(featurizers=[FourCorner()])
result = featurizer.extract('明天')
```

## Companies Using This / 在使用的公司

<img src="image/huya_tv.png" align="left" height="48">

<br/>
<br/>

## TODO

* Add Unicode IDS representation from iQIYI's FASPell model / 增加 Unicode 的 IDS 表征，来自爱奇艺 FASPell 模型

## License

MIT
