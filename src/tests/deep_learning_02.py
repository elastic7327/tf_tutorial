import tensorflow as tf
import numpy as np
import pytest

from src.tests.base import TensorFlowTestBase
# logging.warning('Watch out!')  # will print a message to the console
# logging.info('I told you so')  # will not print anything

class TestBasicNNetwork(TensorFlowTestBase):

    @pytest.mark.skip(reason="skip it for a moment")
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

    @pytest.mark.skip(reason="skip it for a moment")
    def other_basic_exmaple(self):
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, [1], name="x")
            b = tf.constant(1.0)
            y = x * b
            x_in = [2]
            y_out = sess.run([y], {x: x_in})

            print(y_out)

    @pytest.mark.skip(reason="skip it for a moment")
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

    @pytest.mark.skip(reason="skip it for a moment")
    def test_convert_to_tensor_test(self):
        with self.test_session() as sess:
            tendor_3d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
            tendor_3d = tf.convert_to_tensor(tendor_3d, dtype=tf.float64)
            print(tendor_3d.get_shape())
            print(sess.run(tendor_3d))

    @pytest.mark.skip(reason="skip it for a moment")
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

    def test_go_get_tensor(self):
        constant_A = tf.constant([100.0])
        constant_B = tf.constant([300.0])
        constant_C = tf.constant([3.0])

        sum_ = tf.add(constant_A, constant_B)
        mul_ = tf.multiply(constant_A, constant_C)

        with self.test_session() as sess:
            result = sess.run([sum_, mul_])
            print(result)

    @pytest.mark.skip(reason="skip it for a moment")
    def test_basic_feed_example(self):
        a = 3
        b = 3

        x = tf.placeholder(tf.float32, shape=(a, b))
        y = tf.add(x, x)

        data = np.random.rand(a, b)

        with self.test_session() as sess:
            print(sess.run(y, feed_dict={x: data}))

    def test_read_exported_pb_file(self):
        import os
        fm = "frozen_model.pb"
        MODEL_PATH = os.getcwd() + "/src/tests/" + fm
        # Read the graph definition file
        with open(MODEL_PATH, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

                # Load the graph stored in `graph_def` into `graph`
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

        # Enforce that no new nodes are added
        graph.finalize()

        # input_x = tf.constant([0.5182681243, 0.7029702970297029, 42.88, 0.04271770496])
        # out_y = tf.placeholder(tf.float32, shape=[1, 3], name="pred")
        input_op = graph.get_operation_by_name('place_x_data')
        input_tensor = input_op.outputs[0]

        out_op = graph.get_operation_by_name('model_output/output')
        output_tensor = out_op.outputs[0]


        is_training_op = graph.get_operation_by_name('is_training')
        is_training_tensor = is_training_op.outputs[0]

        x = np.array([[0.5182681243, 0.7029702970297029, 42.88, 0.04271770496]]) - 5

        sess_config = tf.ConfigProto(
                    log_device_placement=False,
                    allow_soft_placement = True,
                    gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=1
                    )
                )

        with self.test_session(graph=graph, config=sess_config) as sess:
            # pred = sess.run([out_y], feed_dict={x_data: input_x, is_training: False})
            preds = sess.run(output_tensor, {input_tensor : x, is_training_tensor: False})
            print(preds)


    @pytest.mark.skip(reason="skip it for a moment")
    def test_tensor_board(self):
        input_value = tf.constant(0.5, name="input_value")
        weight = tf.Variable(1.0, name="weight")
        expected_output = tf.constant(0.0, name="expected_output")
        model = tf.multiply(input_value, weight, name="model")
        # mul_ = tf.multiply(constant_A, constant_C)
        loss_function = (model - expected_output)**2 # 보통 이렇게 제곱을 해서 구함 손실률

        optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)

        summaries = tf.merge_all_summaries()
        summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

        with self.test_session() as sess:
            pass
