import tensorflow as tf

import hanzi_char_featurizer

feature = hanzi_char_featurizer.featurize_as_tensor('usage/data.txt')

with tf.Session() as sess:
    sess.run(tf.initializers.tables_initializer())
    for _ in range(3):
        print('-' * 20)
        data = sess.run(feature)
        print(data)
