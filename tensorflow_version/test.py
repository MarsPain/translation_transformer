import numpy as np
import pandas as pd
import tensorflow as tf


# a = np.asarray([[1, 1], [1, 1], [1, 1]])
# b = a * 2
# print(b, b.shape)
#
# a = [0, 1, 2, 3, 4, 5]
# print(a[:-1])

# a = tf.Variable([[0, 1, 2, 3], [0, 1, 2, 3]], name="a")
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(a.eval(session=sess))
#     b = tf.ones_like(a[:, :1])*2
#     print(b.eval(session=sess))
#     c = tf.concat((b, a[:, :-1]), -1)
#     print(c.eval(session=sess))
