from hanzi_chaizi import HanziChaizi


class ChaiZi(object):
    def __init__(self, params=None):
        self.params = params if params else {}
        self.hc = HanziChaizi(**self.params)

    def extract(self, char_seq):
        result = [self.hc.query(i) for i in char_seq]
        return result


if __name__ == "__main__":
    obj = ChaiZi()
    result = obj.extract('明天一定龖麣')

    print(result)
