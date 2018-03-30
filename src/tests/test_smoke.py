import tensorflow as tf

from src.tests.base import TensorFlowTestBase

import pytest

class SquareTest(TensorFlowTestBase):

    def test_square(self):
        with self.test_session():
            x = tf.square([2, 3])
            self.assertAllEqual(x.eval(), [4, 9])

    def test_hello_tensorflow(self):
        with self.test_session() as sess:
            hello = tf.constant("Hello, TensorFlow!")
            print(sess.run(hello))

    def test_add_tensors(self):
        with self.test_session() as sess:
            a = tf.constant(10)
            b = tf.constant(32)

            c = tf.add(a, b)

            print(sess.run(c))  # 이런식으로 지연실행을 한다. 원하는 시점에 실행을 할 수있다.
            print(sess.run([a, b, c])) # 분명 리스트를 반환 해줍니다.

