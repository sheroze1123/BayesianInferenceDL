from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from dolfin import set_log_level
from generate_fin_dataset import generate_five_param, FinInput
from models.simple_dnn_five_param import simple_dnn as model

set_log_level(40)

def main(argv):
    '''
    Sets up inputs to the estimator and trains it. 
    Evaluates the trained model on a validation set and returns
    the RMSE.
    '''
    batch_size  = 10
    eval_size   = 1000
    eval_steps  = 1000
    res         = 40
    train_size  = 1000
    train_steps = 5000

    t_i = time.time()

    # This class performs forward solves for the thermal fin problem
    finInstance = FinInput(batch_size, res) 

    # Generate training set
    train_set = generate_five_param(train_size, res)

    def train_fn():
        return (train_set.shuffle(train_steps).batch(batch_size).repeat().make_one_shot_iterator().get_next())

    #############################################################
    # Training
    #############################################################

    config = tf.estimator.RunConfig(save_summary_steps=10, model_dir='data_rom_error_five_param_new')

    regressor = tf.estimator.Estimator(
                            config   = config,
                            model_fn = model,
                            params   = {"learning_rate" : 1.00,
                                        "batch_size"    : batch_size,
                                        "optimizer"     : tf.train.AdadeltaOptimizer})

    regressor.train(input_fn=train_fn, steps=train_steps)

    t_f = time.time()
    print("Training time taken: {} sec".format(t_f - t_i))

    #############################################################
    # Testing
    #############################################################

    test_set = generate_five_param(eval_size, res)  
    def eval_fn():
        return (test_set.shuffle(eval_size).batch(batch_size).repeat().make_one_shot_iterator().get_next())

    # TODO Add more interesting metrics to check during evaluation time
    #  logging_hook = tf.train.LoggingTensorHook(
        #  tensors={"loss_c": "l2_loss"}, every_n_iter=5)

    eval_result = regressor.evaluate(input_fn=eval_fn, steps=eval_steps)
    print("RMSE for test set: {}".format(eval_result["rmse"]))

    print("Relative Error for test set: {}".format(eval_result["rel_err"]))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
