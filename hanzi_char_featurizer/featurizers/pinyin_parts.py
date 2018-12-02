import tensorflow as tf

from pypinyin import pinyin, Style
from pypinyin.style._constants import _INITIALS


raw_FINALS = 'i,u,v,a,ia,ua,o,uo,e,ie,ve,ai,uai,ei,uei,ao,iao,ou,iou,an,ian,uan,van,en,in,uen,vn,ang,iang,uang,eng,ing,ueng,ong,iong'.split(',')

# ordered by length, so FINALS can work with endswith correctly
FINALS = sorted(raw_FINALS, key=lambda x: len(x), reverse=True)


class PinYinParts(object):
    common_params = {
        'heteronym': True,
        'errors': lambda x: [i for i in x]  # return char literally
    }

    padding = ('-', '-', '0')

    def __init__(self, params=None):
        self.params = params if params else {'pad_len': 1}

    def extract_raw(self, char_seq):
        char_pinyin_seq = pinyin(char_seq, style=Style.TONE3, **self.common_params)

        initial_result = []
        final_result = []
        tone_result = []

        for char_pinyin in char_pinyin_seq:
            char_initial = []
            char_final = []
            char_tone = []
            for candidate_pinyin in char_pinyin:
                if candidate_pinyin == '-':
                    char_initial.append('-')
                    char_final.append('-')
                    char_tone.append('0')

                    continue

                initial = '-'
                # _INITIALS has a special order can work with startswith() correctly
                for i in _INITIALS:
                    if candidate_pinyin.startswith(i):
                        initial = i
                        break
                char_initial.append(initial)

                # tone part
                tone = candidate_pinyin[-1]
                char_tone.append(tone)

                # final part
                candidate_pinyin_without_tone = candidate_pinyin[:-1]
                final = '-'
                # FINALS has a special order can work with endswith() correctly
                for i in FINALS:
                    if candidate_pinyin_without_tone.endswith(i):
                        final = i
                        break
                char_final.append(final)

            initial_result.append(char_initial)
            final_result.append(char_final)
            tone_result.append(char_tone)

        # zip parts together
        result = []
        for char_index in range(len(initial_result)):
            item = list(
                zip(
                    initial_result[char_index],
                    final_result[char_index],
                    tone_result[char_index]
                )
            )
            result.append(item)

        # padding parts
        padded_result = []
        for item in result:
            padding_length = max((self.params['pad_len'] - len(item), 0))
            padding_part = [self.padding] * padding_length
            after_pading = item + padding_part

            # make sure no longer than this
            fine_item = after_pading[:self.params['pad_len']]

            padded_result.append(fine_item)

        return padded_result

    def extract(self, char_seq):
        padded_result = self.extract_raw(char_seq)

        feature_column_data = {
            'initial': [],  # 声母 in Chinese
            'final': [],  # 韵母 in Chinese
            'tone': []  # 声调 in Chinese
        }

        for char_feature in padded_result:
            initial_feature = []
            final_feature = []
            tone_feature = []

            for candiate_feature in char_feature:
                initial_feature.append(candiate_feature[0])
                final_feature.append(candiate_feature[1])
                tone_feature.append(candiate_feature[2])

            feature_column_data['initial'].append(initial_feature)
            feature_column_data['final'].append(final_feature)
            feature_column_data['tone'].append(tone_feature)

        return tuple(feature_column_data[i] for i in ['initial', 'final', 'tone'])

    @staticmethod
    def get_vocabulary():
        return _INITIALS, FINALS, [str(i) for i in (range(5))]

    @classmethod
    def get_data_type(cls):
        return (tf.string, ) * 3

    @classmethod
    def get_data_shape(cls):
        return (tf.TensorShape([None, None]), ) * 3


if __name__ == "__main__":
    obj = PinYinParts()
    # res = obj.extract_raw('衣服熊无雨！😁月')
    # print(res)

    res = obj.extract('行')
    print(res)


    # print(PinYinParts.get_vocabulary())
