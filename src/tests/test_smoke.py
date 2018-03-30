import tensorflow as tf

from src.tests.base import TensorFlowTestBase

import pytest

import logging
# logging.warning('Watch out!')  # will print a message to the console
# logging.info('I told you so')  # will not print anything

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


    @pytest.mark.skip(reason="skip it for a moment")
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

    def test_linear_regression(self):
        with self.test_session() as sess:
            x_data = [1, 2, 3]
            y_data = [1, 2, 3]

            # -1.0 부터 1.0 사이의 균등분포 를 가진 무작위 값으로 초기화
            W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
            b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

            X = tf.placeholder(tf.float32, name="X")
            Y = tf.placeholder(tf.float32, name="Y")

            # Hypothesis 예측값
            # W: Weight: 가중치
            # X: X 값
            # b: Bias: 편향

            hypothesis = W * X + b # 예측값은 => 가중치 곱하기 미지수에 편향(바이어스)를 더한값이다.

            # 함수 그대로 영어로 해석해보자, 나중에 깊이 공부를 하게 되는경우 이게 매우 중요할것같다.
            # 비용 => 예측값과 실제값을 뺀뒤에 제곱을해서, 평균을 내어구한다.
            cost = tf.reduce_mean(tf.square(hypothesis - Y))
            # learning_rate 값에 따라서 학습 시간이 매우 차이가 나기때문에, 적절한 값을 삽입해주는것이 중요
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            train_op = optimizer.minimize(cost)

            sess.run(tf.global_variables_initializer())

            # 최적화를 수행하는 그래프인 train_op를 실행하고, 실행시 마다 변화하는 손실값을 출력하는코드

            # 최적화 함수란, 가중치와 편향값 Weight, bias 값을 변경해가면서 손실값을 최소화하는 가장 최적화된 가중치
            # 와 편향값을 찾아주는 함수이다.!
            # 이때 값들을 무작위로 변경하면 시간이 너무 오래걸리고 학습 시간도 예측하기 어려울것이다. 항상 적.절.하.게

            for step in range(100):
                _, cost_val = sess.run(
                        [train_op, cost],
                        feed_dict={
                            X: x_data,
                            Y: y_data,
                            }
                        )
                print(step, cost_val, sess.run(W), sess.run(b))

            print("X: 5, Y: {}", sess.run(hypothesis, feed_dict={X: 5}))
            print("X: 2.5, Y: {}", sess.run(hypothesis, feed_dict={X: 2.5}))
