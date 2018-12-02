from pypinyin import pinyin, Style
from pypinyin.style.bopomofo import BOPOMOFO_TABLE


class PinYin(object):
    default_params = {
        'style': Style.BOPOMOFO,
        'errors': lambda x: [i for i in x]  # return char literally
    }

    def __init__(self, params=None):
        self.params = params if params else self.default_params

    def extract(self, char_seq):
        raw_result = pinyin(char_seq, **self.params)
        result = [i[0] for i in raw_result]

        return result

    def get_vocabulary(self):
        return BOPOMOFO_TABLE.values()


if __name__ == "__main__":
    obj = PinYin()
    res = obj.extract('明天天气真好！😁')
    print(res)
