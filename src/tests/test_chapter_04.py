import tensorflow as tf
import numpy as np
import pytest

from src.tests.base import TensorFlowTestBase
# logging.warning('Watch out!')  # will print a message to the console
# logging.info('I told you so')  # will not print anything

class TestBasicNNetwork(TensorFlowTestBase):

    def test_basic_network(self):
        with self.test_session() as sess:
            # [털, 날개]
            # 있으면 1, 없으면 0
            x_data = np.array(
                [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]
                )
