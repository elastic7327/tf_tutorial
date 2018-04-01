import tensorflow as tf
import numpy as np
import pytest

from src.tests.base import TensorFlowTestBase
# logging.warning('Watch out!')  # will print a message to the console
# logging.info('I told you so')  # will not print anything

class TestBasicNNetwork(TensorFlowTestBase):

    def test_basic_network(self):
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, [1], name="x")
            y = tf.placeholder(tf.float32, [1], name="y")
            z = tf.constant(2.0)
            y = x * z
            x_in = [100]
            y_output =sess.run(y, {x: x_in})

            print(y_output)

            print(y_output)

    def other_basic_exmaple(self):
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, [1], name="x")
            b = tf.constant(1.0)
            y = x * b
            x_in = [2]
            y_out = sess.run([y], {x: x_in})

            print(y_out)

    def test_what_is_rank(self):
        with self.test_session() as sess:
            scalar = tf.constant(100)
            vector = tf.constant([1, 2, 3, 4, 5])
            matrix = tf.constant([[1 ,2, 3], [4 ,5 ,6]])
            cube_matrix = tf.constant([[[1], [2], [3]], [[4], [5], [6]]])

            print(scalar.get_shape())
            print(vector.get_shape())
            print(matrix.get_shape())
            print(cube_matrix.get_shape())
            print(sess.run(cube_matrix))

            tendor_1d = np.array([1, 2, 3, 4, 5, 6, 7])
            tendor_1d = tf.constant(tendor_1d)
            print(tendor_1d.get_shape())
            print(sess.run(tendor_1d))

    def test_convert_to_tensor_test(self):
        with self.test_session() as sess:
            tendor_3d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

            tendor_3d = tf.convert_to_tensor(tendor_3d, dtype=tf.float64)

            print(tendor_3d.get_shape())
            print(sess.run(tendor_3d))

    def test_update_value_test(self):
        one = tf.constant(1)
        value = tf.Variable(0, name="value")
        new_value = tf.add(value, one)
        update_value = tf.assign(value, new_value)
        init_var = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_var)
            print(sess.run(value))

            for _ in range(10):
                sess.run(update_value)
                print(sess.run(value))
