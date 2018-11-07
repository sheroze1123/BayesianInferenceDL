from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from dolfin import *
from forward_solve import Fin
from generate_fin_dataset import generate, FinInput, load_saved_dataset
from models.simple_dnn import simple_dnn as model

set_log_level(40)

def pred_dataset(finInstance):
    '''
    Creates examples to use the DNN error predictor
    '''
    pred_x = np.zeros((2, finInstance.dofs))
    #  m = interpolate(Expression("2*x[0] + 3*x[1] + 1.5", degree=2), finInstance.V)
    m = interpolate(Expression("1 + sin(x[0])* sin(x[0])", degree=2), finInstance.V)

    w, y, A, B, C = finInstance.solver.forward(m)
    pred_x[0][:] = m.vector()[:]
    psi = np.dot(A, finInstance.phi)
    A_r, B_r, C_r, x_r, y_r = finInstance.solver.reduced_forward(A, B, C, psi, finInstance.phi)
    error_1 = y - y_r

    m = interpolate(Expression("2*x[0] + 0.1", degree=2), finInstance.V)
    w, y, A, B, C = finInstance.solver.forward(m)
    pred_x[1][:] = m.vector()[:]
    psi = np.dot(A, finInstance.phi)
    A_r, B_r, C_r, x_r, y_r = finInstance.solver.reduced_forward(A, B, C, psi, finInstance.phi)
    error_2 = y - y_r

    return y, y_r, error_2, tf.data.Dataset.from_tensor_slices(pred_x)


def main(argv):
    '''
    Sets up inputs to the estimator and trains it. 
    Evaluates the trained model on a validation set and performs 
    proof of concept of prediction for a given parameter.
    '''
    batch_size  = 1
    eval_size   = 100
    eval_steps  = 100
    res         = 40
    train_size  = 200
    train_steps = 200

    #############################################################
    # Set up input functions
    #############################################################

    # This class performs forward solves for the thermal fin problem
    finInstance = FinInput(batch_size, res) 

    # Generate test set and training set
    #  train_set = generate(train_size, res)
    train_set = load_saved_dataset()
    test_set = generate(eval_size, res)  # Randomly generate validation set per run
    y, y_r, error_2, pred_set = pred_dataset(finInstance)

    def train_fn():
        return (train_set.shuffle(train_size).batch(batch_size).repeat().make_one_shot_iterator().get_next())
    def eval_fn():
        return (test_set.shuffle(eval_size).batch(batch_size).repeat().make_one_shot_iterator().get_next())
    def pred_fn():
        return (pred_set.shuffle(10).batch(2).make_one_shot_iterator().get_next())

    # TODO Add more interesting metrics to check during evaluation time
    #  logging_hook = tf.train.LoggingTensorHook(
        #  tensors={"loss_c": "l2_loss"}, every_n_iter=5)


    #############################################################
    # Set up estimator
    #############################################################

    config = tf.estimator.RunConfig(save_summary_steps=100, model_dir='data_rom_error')

    regressor = tf.estimator.Estimator(
        config = config,
        model_fn=model, 
        params={"num_nodes":finInstance.dofs,
                "learning_rate":1.00,
                "batch_size":batch_size,
                "optimizer":tf.train.AdadeltaOptimizer})

    regressor.train(input_fn=train_fn, steps=train_steps)
    #  regressor.train(input_fn=finInstance.train_input_fn, steps=train_steps)
                            #  steps=train_steps, hooks=[logging_hook])

    eval_result = regressor.evaluate(input_fn=eval_fn, steps=eval_steps)
    print("RMSE for test set: {}".format(eval_result["rmse"]))

    prediction = list(regressor.predict(input_fn=pred_fn))
    print(prediction)
    print("y = {}, y_r = {}, e_pred = {}, e_true = {}".format(y, y_r, prediction[1][0], error_2))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.app.run(main=main)
