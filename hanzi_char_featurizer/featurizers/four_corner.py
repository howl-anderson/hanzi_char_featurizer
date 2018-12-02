import tensorflow as tf

from four_corner_method import FourCornerMethod


class FourCorner(object):
    def __init__(self, params=None):
        self.params = params if params else {}
        self.fcm = FourCornerMethod(**self.params)

    def extract_raw(self, char_seq):
        result = [[i for i in self.fcm.query(i)] for i in char_seq]
        return result

    def extract(self, char_seq):
        raw_result = self.extract_raw(char_seq)

        result = tuple(zip(*raw_result))

        return result

    @staticmethod
    def get_vocabulary():
        def range_str(x):
            return [str(i) for i in (range(x))]

        return range_str(10), range_str(10), range_str(10), range_str(10), range_str(5)

    @classmethod
    def get_data_type(cls):
        return (tf.string,) * 5

    @classmethod
    def get_data_shape(cls):
        return (tf.TensorShape([None]), ) * 5


if __name__ == "__main__":
    obj = FourCorner()
    result = obj.extract('明天')
    print(result)
