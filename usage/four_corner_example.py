import tensorflow as tf

from hanzi_char_featurizer.featurizers.four_corner import FourCorner


def gen():
    obj = FourCorner()

    with open('data.txt', 'rt') as fd:
        for line in fd:
            res = obj.extract(line.strip())
            yield res


dataset = tf.data.Dataset.from_generator(
    gen,
    (tf.string,) * 5,
    (tf.TensorShape([None]),) * 5
)

value = dataset.make_one_shot_iterator().get_next()

data = {
    'upper_left': value[0],  # 左上
    'upper_right': value[1],  # 右上
    'lower_left': value[2],  # 左下
    'lower_right': value[3],  # 右下
    'middle_right': value[4]  # 补码
}

vocabulary = FourCorner.get_vocabulary()

upper_left_vocabulary = vocabulary[0]
upper_right_vocabulary = vocabulary[1]
lower_left_vocabulary = vocabulary[2]
lower_right_vocabulary = vocabulary[3]
middle_right_vocabulary = vocabulary[4]

upper_left_feature_column = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='upper_left',
        vocabulary_list=upper_left_vocabulary)
)

upper_right_feature_column = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='upper_right',
        vocabulary_list=upper_right_vocabulary)
)

lower_left_feature_column = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='lower_left',
        vocabulary_list=lower_left_vocabulary)
)

lower_right_feature_column = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='lower_right',
        vocabulary_list=lower_right_vocabulary)
)

middle_right_feature_column = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='middle_right',
        vocabulary_list=middle_right_vocabulary)
)

feature = tf.feature_column.input_layer(
    data,
    [
        upper_left_feature_column,
        upper_right_feature_column,
        lower_left_feature_column,
        lower_right_feature_column,
        middle_right_feature_column
    ]
)
# feature = tf.feature_column.input_layer(data, [upper_left_feature_column])
# feature = tf.feature_column.input_layer(data, [middle_right_feature_column])

with tf.Session() as sess:
    sess.run(tf.initializers.tables_initializer())
    for _ in range(1):
        print('-' * 20)
        # print(sess.run(value))
        # print(sess.run((data['initial'], data['final'], data['tone'])))
        data = sess.run((feature,
                         data['upper_left'], data['upper_right'],
                         data['lower_left'], data['lower_right'],
                         data['middle_right']))
        print(data)
