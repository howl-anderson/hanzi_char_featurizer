import tensorflow as tf

from hanzi_char_featurizer.featurizers.pinyin_parts import PinYinParts


def gen():
    obj = PinYinParts()

    with open('data.txt', 'rt') as fd:
        for line in fd:
            res = obj.extract(line.strip())
            yield res


dataset = tf.data.Dataset.from_generator(
    gen,
    (tf.string, ) * 3,
    (tf.TensorShape([None, None]), ) * 3
)

value = dataset.make_one_shot_iterator().get_next()

data = {
    'initial': value[0],  # 声母 in Chinese
    'final': value[1],  # 韵母 in Chinese
    'tone': value[2]  # 声调 in Chinese
}

vocabulary = PinYinParts.get_vocabulary()

initial_vocabulary = vocabulary[0]
final_vocabulary = vocabulary[1]
tone_vocabulary = vocabulary[2]

initial_feature_column = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='initial',
        vocabulary_list=initial_vocabulary)
)

final_feature_column = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='final',
        vocabulary_list=final_vocabulary)
)

tone_feature_column = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='tone',
        vocabulary_list=tone_vocabulary)
)

feature = tf.feature_column.input_layer(data, [initial_feature_column, final_feature_column, tone_feature_column])
# feature = tf.feature_column.input_layer(data, [initial_feature_column])
# feature = tf.feature_column.input_layer(data, [final_feature_column])
# feature = tf.feature_column.input_layer(data, [tone_feature_column])

with tf.Session() as sess:
    sess.run(tf.initializers.tables_initializer())
    for _ in range(1):
        print('-' * 20)
        # print(sess.run(value))
        # print(sess.run((data['initial'], data['final'], data['tone'])))
        data = sess.run((feature, data['initial'], data['final'], data['tone']))
        print(data)
