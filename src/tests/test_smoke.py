import tensorflow as tf

from src.tests.base import TensorFlowTestBase

class SquareTest(TensorFlowTestBase):
    def testSquare(self):
        with self.test_session():
            x = tf.square([2, 3])
            self.assertAllEqual(x.eval(), [4, 9])
