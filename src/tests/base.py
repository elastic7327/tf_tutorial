import tensorflow as tf

class TensorFlowTestBase(tf.test.TestCase):

    def setUp(self):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # out put에 컬러링 넣기위해서
        self.W  = '\033[0m'  # white (normal)
        self.R  = '\033[31m' # red
        self.G  = '\033[32m' # green
        self.O  = '\033[33m' # orange
        self.B  = '\033[34m' # blue
        self.P  = '\033[35m' # purple

    def tearDown(self):
        pass
