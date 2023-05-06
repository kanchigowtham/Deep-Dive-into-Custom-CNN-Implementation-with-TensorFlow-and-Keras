# Kanchi, Gowtham Kumar
# 1002-044-003
# 2022_11_13
# Assignment-04-03
import pytest
import numpy as np
from cnn import CNN
import os


def test():
    from tensorflow.keras.datasets import mnist

    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_test = y_test.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    number_of_samples_to_use_for_training = 900
    number_of_samples_to_use_for_testing = 250
    X_train = X_train[indices[:number_of_samples_to_use_for_training]]
    y_train = y_train[indices[:number_of_samples_to_use_for_training]]
    X_test = X_test[indices[:number_of_samples_to_use_for_testing]]
    y_test = y_test[indices[:number_of_samples_to_use_for_testing]]
    cnn = CNN()
    cnn.add_input_layer(shape=input_dimension, name="input0")
    cnn.append_dense_layer(num_nodes=64, activation='relu', name="first")
    weights_1 = cnn.get_weights_without_biases(layer_name="first")
    weights_set = np.full_like(weights_1, 2)
    cnn.set_weights_without_biases(weights_set, layer_name="first")
    bias_1 = cnn.get_biases(layer_name="first")
    cnn.append_dense_layer(num_nodes=10, activation='linear', name="second")
    weights_2 = cnn.get_weights_without_biases(layer_name="second")
    weights_set = np.full_like(weights_2, 2)
    cnn.set_weights_without_biases(weights_set, layer_name="second")
    bias_2 = cnn.get_biases(layer_name="second")
    bias = np.full_like(bias_1, 2)
    bias[0] = bias[0] * 2
    bias2 = np.full_like(bias_2, 2)
    bias2[0] = bias2[0] * 2
    cnn.set_metric("accuracy")
    cnn.set_loss_function("SparseCategoricalCrossentropy")
    cnn.set_optimizer("SGD")
    cnn.set_biases(bias, layer_name="first")
    cnn.set_biases(bias2, layer_name="second")
    actual = cnn.train(X_train, y_train, 100, 2)
    evaluate = cnn.evaluate(X_test, y_test)
    print(actual)
    print(evaluate)
    np.testing.assert_almost_equal(actual, np.array(
        [2.3025, 2.3025]), decimal=4)
    np.testing.assert_almost_equal(
         evaluate, np.array([2.3026, 0.076]), decimal=4)
