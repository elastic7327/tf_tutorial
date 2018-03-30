import tensorflow as tf

class TensorFlowTestBase(tf.test.TestCase):

    def setUp(self):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def tearDown(self):
        pass
