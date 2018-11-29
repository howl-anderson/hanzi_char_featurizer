from hanzi_char_featurizer.featurizers.chaizi import ChaiZi
from hanzi_char_featurizer.featurizers.four_corner import FourCorner
from hanzi_char_featurizer.featurizers.pinyin import PinYin

def featurize(char_seq, featurizers=None):
    if not featurizers:
        featurizers = [PinYin(), FourCorner(), ChaiZi()]

    featurize_result = []
    for featurizer in featurizers:
        featurize_result.append(featurizer.extract(char_seq))

    result = list(zip(*featurize_result))

    return result


if __name__ == "__main__":
    result = featurize('明')
    print(result)

    result = featurize('明天')
    print(result)