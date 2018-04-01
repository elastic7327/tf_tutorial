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

