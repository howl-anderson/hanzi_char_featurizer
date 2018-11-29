from four_corner_method import FourCornerMethod


class FourCorner(object):
    def __init__(self, params=None):
        self.params = params if params else {}
        self.fcm = FourCornerMethod(**self.params)

    def extract(self, char_seq):
        result = [self.fcm.query(i) for i in char_seq]
        return result
