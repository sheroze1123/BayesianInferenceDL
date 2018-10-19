from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pandas import DataFrame, read_csv
import pdb
from numpy.random import rand
from forward_solve import *
from fin_functionspace import get_space

class FinInput:
    def __init__(self, batch_size, resolution):
        self.resolution = resolution
        self.V = get_space(resolution)
        self.dofs = len(self.V.dofmap().dofs())
        self.phi = np.loadtxt('basis.txt',delimiter=",")
        self.batch_size = batch_size

    def train_input_fn(self):
        params = np.random.uniform(0.1, 1, (self.batch_size, self.dofs))
        errors = np.zeros((self.batch_size, 1))

        for i in range(self.batch_size):
            m = Function(self.V)
            m.vector().set_local(params[i,:])
            w, y, A, B, C, dA_dz = forward(m, self.V)
            psi = np.dot(A, self.phi)
            A_r, B_r, C_r, x_r, y_r = reduced_forward(A, B, C, psi, self.phi)
            errors[i][0] = y - y_r 

        return ({'x':tf.convert_to_tensor(params)}, tf.convert_to_tensor(errors))

    def eval_input_fn(self):
        params = np.random.uniform(0.1, 1, (self.batch_size, self.dofs))
        errors = np.zeros((self.batch_size, 1))

        for i in range(self.batch_size):
            m = Function(self.V)
            m.vector().set_local(params[i,:])
            w, y, A, B, C, dA_dz = forward(m, self.V)
            psi = np.dot(A, self.phi)
            A_r, B_r, C_r, x_r, y_r = reduced_forward(A, B, C, psi, self.phi)
            errors[i][0] = y - y_r 

        return ({'x':tf.convert_to_tensor(params)}, tf.convert_to_tensor(errors))


def main(argv):
    batch_size  = tf.flags.FLAGS.batch_size
    train_steps = tf.flags.FLAGS.train_steps
    eval_steps  = tf.flags.FLAGS.eval_steps

    config = tf.estimator.RunConfig(save_summary_steps=5, model_dir='data_rom_error')

    finInstance = FinInput(batch_size, 40) 

    #  logging_hook = tf.train.LoggingTensorHook(
        #  tensors={"loss_c": "l2_loss"}, every_n_iter=5)

    regressor = tf.estimator.Estimator(
        config = config,
        model_fn=dnn_model, 
        params={"nodes":finInstance.dofs})

    regressor.train(input_fn=finInstance.train_input_fn, steps=train_steps)
                            #  steps=train_steps, hooks=[logging_hook])

    eval_result = regressor.evaluate(input_fn=finInstance.eval_input_fn, steps=eval_steps)
    print(eval_result)

    m = interpolate(Expression("2*x[0] + 3*x[1] + 1.5", degree=2), finInstance.V)
    w, y, A, B, C, dA_dz = forward(m, finInstance.V)
    psi = np.dot(A, finInstance.phi)
    A_r, B_r, C_r, x_r, y_r = reduced_forward(A, B, C, psi, finInstance.phi)
    error = y - y_r

    #  pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            #  x={'x':np.array([m.vector()[:]])}, 
            #  shuffle=False)
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x':np.array([m.vector()[:]])}, 
            shuffle=False)

    prediction = list(regressor.predict(input_fn=pred_input_fn))
    #  import pdb; pdb.set_trace()
    print("y = {}, y_r = {}, e_pred = {}, e_true = {}".format(y, y_r, prediction[0][0], error))

def dnn_model(features, labels, mode, params):
    '''
    Deep Neural Network to map nodal values of temperature to error
    between full model and the reduced order model

    Arguments:
        features - This is batch_features from input_fn
        labels   - This is batch_labels from input_fn
        mode     - An instance of tf.estimator.ModeKeys, see below
        params   - Additional configuration
    '''

    batch_size = tf.flags.FLAGS.batch_size

    # input_layer shape = [batch_size, height, width, channels]
    # -1 for batch size, which specifies that this dimension should be dynamically
    # computed based on the number of input values in features["x"]
    dense1 = tf.layers.dense(features["x"], units=params["nodes"], activation=tf.nn.relu)

    dense2 = tf.layers.dense(dense1, units=params["nodes"], activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.02, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense3 = tf.layers.dense(dropout2, units=params["nodes"], activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(
        inputs=dense3, rate=0.02, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout3, units=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)
    
    # TODO: rewrite this with forwad solve loss
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, logits)
    #  loss = tf.reduce_mean(tf.squared_difference(
        #  labels, logits), name='l2_loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.2)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # TODO: Per param error values in the metrics

    # Add evaluation metrics (for EVAL mode)
    #  eval_metric_ops = {
        #  "accuracy": tf.metrics.accuracy(
        #  labels=labels, predictions=logits)}
    eval_metric_ops = {}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.flags.DEFINE_integer('batch_size', 20, 'Number of images to process in a batch.')
    tf.flags.DEFINE_integer('train_steps', 1000, 'Number of training steps to take.')
    tf.flags.DEFINE_integer('eval_steps', 100, 'Number of evaluation steps to take.')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
