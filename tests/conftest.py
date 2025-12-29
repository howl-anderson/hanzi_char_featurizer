import pytest

from hanzi_char_featurizer import Featurizer
from hanzi_char_featurizer.featurizers.four_corner import FourCorner
from hanzi_char_featurizer.featurizers.pinyin_parts import PinYinParts
from hanzi_char_featurizer.featurizers.pinyin import PinYin
from hanzi_char_featurizer.featurizers.chaizi import ChaiZi


@pytest.fixture
def featurizer():
    return Featurizer()


@pytest.fixture
def four_corner():
    return FourCorner()


@pytest.fixture
def pinyin_parts():
    return PinYinParts()


@pytest.fixture
def pinyin():
    return PinYin()


@pytest.fixture
def chaizi():
    return ChaiZi()
