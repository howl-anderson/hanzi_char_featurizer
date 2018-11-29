from pypinyin import pinyin


class PinYin(object):
    def __init__(self, params=None):
        self.params = params if params else {}

    def extract(self, char_seq):
        raw_result = pinyin(char_seq, **self.params)
        result = [i[0] for i in raw_result]

        return result
