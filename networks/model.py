import sys
sys.path.append(sys.path[0] + "/..")

import chess
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pychess_utils as chess_utils
from deepmind_mcts import MCTS

# Setup for less verbose TensorFlow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

class ChessNetwork:
    DATA_PATH = "ACZData/self_play.csv"
    MODEL_DIRECTORY = "Model/"
    EXPORT_DIRECTORY = "Export/"
    PIECE_NAMES = ['pawn_', 'knight_', 'bishop_', 'rook_', 'queen_', 'king_']
    RES_BLOCKS = 8

    def __init__(self):
        self.estimator = tf.estimator.Estimator(
            model_dir=self.MODEL_DIRECTORY,
            model_fn=self.model_fn,
            params={}
        )

    def export_model(self):
        serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            features={'x': tf.compat.v1.placeholder(tf.float32, shape=[1, 8*8*13])}
        )
        self.estimator.export_savedmodel(self.EXPORT_DIRECTORY, serving_input_fn)

    def train_model(self, epochs=None, shuffle=True, steps=1):
        self.estimator.train(
            input_fn=self.input_fn(self.DATA_PATH, num_epochs=epochs, shuffle=shuffle),
            steps=steps
        )

    def feature_columns(self):
        columns = []
        for player in range(2):
            for piece in self.PIECE_NAMES:
                for square in range(64):
                    columns.append(f"{player}{piece}{square}")
        columns.extend([f"turn{square}" for square in range(64)])
        return columns

    def target_columns(self):
        return ['probs', 'value']

    def decode_move(self, encoded_labels):
        decoded_labels = []
        labels = encoded_labels.split('#')
        for label in labels:
            move, prob = label.strip('()').split(':')
            decoded_labels.append((int(move), float(prob)))
        return decoded_labels

    def create_policy_array(self, labels):
        policy_array = [0.0] * (8 * 8 * 73)
        for move, prob in labels:
            policy_array[move] = prob
        return np.array(policy_array)

    def input_fn(self, filepath, num_epochs, batch_size=32, shuffle=False, num_threads=4):
        feature_columns = self.feature_columns()
        target_columns = self.target_columns()
        dataset = pd.read_csv(
            tf.io.gfile.GFile(filepath),
            usecols=feature_columns + target_columns,
            engine="python"
        ).dropna()

        dataset['policy'] = dataset.probs.apply(lambda x: self.create_policy_array(self.decode_move(x)))
        value_labels = dataset.value
        dataset = dataset.drop(columns=['probs', 'value'])

        return tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={"x": dataset.to_numpy()},
            y=np.hstack([dataset['policy'].to_numpy(), value_labels.to_numpy().reshape(-1, 1)]),
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle=shuffle,
            num_threads=num_threads
        )

    def custom_conv_layer(self, input_layer, filter_height, filter_width, in_channels, out_channels, stride=[1,1,1,1], name="conv"):
        with tf.compat.v1.variable_scope(name):
            filters = tf.compat.v1.get_variable(
                name+"_filter",
                [filter_height, filter_width, in_channels, out_channels],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.05)
            )
            biases = tf.compat.v1.get_variable(name+"_biases", [out_channels], initializer=tf.compat.v1.constant_initializer(0.0))
            conv = tf.nn.conv2d(input_layer, filters, stride, padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            return pre_activation

    def custom_batch_norm(self, inputs, training=True, name="batch_norm"):
        with tf.compat.v1.variable_scope(name):
            return tf.compat.v1.layers.batch_normalization(inputs=inputs, training=training)

    def custom_relu(self, inputs, name="relu"):
        with tf.compat.v1.variable_scope(name):
            return tf.nn.relu(inputs)

    def model_fn(self, features, labels, mode, params):
        if mode != tf.estimator.ModeKeys.PREDICT:
            policy_labels, value_labels = tf.split(labels, [8*8*73, 1], axis=1)

        input_layer = tf.cast(features["x"], tf.float32)
        board_image = tf.reshape(input_layer, [-1, 8, 8, 13])

        conv1 = self.custom_conv_layer(board_image, 3, 3, 13, 256, name="conv1")
        norm1 = self.custom_batch_norm(conv1, name="norm1")
        relu1 = self.custom_relu(norm1, name="relu1")

        current_input = relu1
        for i in range(self.RES_BLOCKS):
            block_id = str(2 + i)
            conv_block = self.custom_conv_layer(current_input, 3, 3, 256, 256, name="conv" + block_id)
            norm_block = self.custom_batch_norm(conv_block, name="norm" + block_id)
            relu_block = self.custom_relu(norm_block, name="relu" + block_id)
            conv_block2 = self.custom_conv_layer(relu_block, 3, 3, 256, 256, name="2conv" + block_id)
            norm_block2 = self.custom_batch_norm(conv_block2, name="2norm" + block_id)
            residual_block = tf.add(current_input, norm_block2)
            relu_block2 = self.custom_relu(residual_block, name="2relu" + block_id)
            current_input = relu_block2

        policy_conv = self.custom_conv_layer(current_input, 1, 1, 256, 2, name="policy_conv")
        policy_norm = self.custom_batch_norm(policy_conv, name="policy_norm")
        policy_relu = self.custom_relu(policy_norm, name="policy_relu")
        policy_flat = tf.reshape(policy_relu, [-1, 128])
        policy_output = tf.compat.v1.layers.dense(inputs=policy_flat, units=8*8*73, activation=tf.nn.sigmoid)

        value_conv = self.custom_conv_layer(current_input, 1, 1, 256, 1, name="value_conv")
        value_norm = self.custom_batch_norm(value_conv, name="value_norm")
        value_relu = self.custom_relu(value_norm, name="value_relu")
        value_flat = tf.reshape(value_relu, [-1, 64])
        value_hidden = tf.compat.v1.layers.dense(inputs=value_flat, units=256, activation=tf.nn.relu)
        value_output = tf.compat.v1.layers.dense(inputs=value_hidden, units=1, activation=tf.nn.tanh)

        predictions = tf.concat([policy_output, value_output], axis=1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    "policy": tf.estimator.export.PredictOutput({"policy": policy_output}),
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({"policy": policy_output, "value": value_output})
                }
            )

        loss = tf.reduce_mean(
            tf.square(tf.subtract(value_labels, value_output)) +
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=policy_output, labels=policy_labels)
        )

        learning_rate = tf.compat.v1.train.exponential_decay(0.01, tf.compat.v1.train.get_global_step(), 100000, 0.96, staircase=True)
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op
        )
