import tensorflow as tf

from src.tests.base import TensorFlowTestBase

import pytest

class SquareTest(TensorFlowTestBase):

    @pytest.mark.skip(reason="skip it for a moment")
    def test_square(self):
        with self.test_session():
            x = tf.square([2, 3])
            self.assertAllEqual(x.eval(), [4, 9])

    @pytest.mark.skip(reason="skip it for a moment")
    def test_hello_tensorflow(self):
        with self.test_session() as sess:
            hello = tf.constant("Hello, TensorFlow!")
            print(hello)
            print(sess.run(hello))

    @pytest.mark.skip(reason="skip it for a moment")
    def test_add_tensors(self):
        with self.test_session() as sess:
            a = tf.constant(10)
            b = tf.constant(32)

            c = tf.add(a, b)

            print(sess.run(c))  # 이런식으로 지연실행을 한다. 원하는 시점에 실행을 할 수있다.
            print(sess.run([a, b, c])) # 분명 리스트를 반환 해줍니다.


    def test_placeholder(self):

        with self.test_session() as sess:
            X = tf.placeholder(tf.float32, [None, 3])
            # 위의 텐서.. 즉 플레이스홀더로 만들어진, 변수에 들어갈 값들은 이런식으로 나와야합니다.
            x_data = [
                    [1, 2, 3],
                    [4, 5, 6]]  # 2 X 3 행렬

            W = tf.Variable(tf.random_normal([3, 2])) # 3행 2열짜리 행렬 생성
            b = tf.Variable(tf.random_normal([2, 1])) # 2행 1 열짜리 행렬 생성

            expr = tf.matmul(X, W) + b
            sess.run(tf.global_variables_initializer())

            print(expr)
            print(x_data)
            print(sess.run(W))
            print(sess.run(b))

            print(sess.run(expr, feed_dict={X: x_data}))
