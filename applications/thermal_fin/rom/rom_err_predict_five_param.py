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
    batch_size        = 10
    eval_dataset_size = 200
    n_tr_epochs       = 300
    res               = 40
    tr_dataset_size   = 1500
    shfl_buf_size     = tr_dataset_size * 4

    t_i = time.time()

    # This class performs forward solves for the thermal fin problem
    finInstance = FinInput(batch_size, res) 

    # Generate training set
    train_set = generate_five_param(tr_dataset_size, res)

    def train_fn():
        return (train_set
                .shuffle(shfl_buf_size)
                .repeat(n_tr_epochs)
                .batch(batch_size)
                .make_one_shot_iterator()
                .get_next())
    train_spec = tf.estimator.TrainSpec(
            input_fn = train_fn, 
            max_steps = tr_dataset_size * n_tr_epochs)

    #############################################################
    # Training
    #############################################################

    config = tf.estimator.RunConfig(save_summary_steps=1, 
            model_dir='data/five_param')

    #  params   = {"learning_rate" : 1.00,
                #  "batch_size"    : batch_size,
                #  "optimizer"     : tf.train.AdadeltaOptimizer}
    params   = {"learning_rate" : 0.01,
                "batch_size"    : batch_size,
                "optimizer"     : tf.train.AdamOptimizer}
    #  params   = {"learning_rate" : 1.0,
                #  "batch_size"    : batch_size,
                #  "optimizer"     : tf.train.GradientDescentOptimizer}

    regressor = tf.estimator.Estimator(config = config, model_fn = model, params = params)

    #  regressor.train(input_fn=train_fn, steps=tr_dataset_size * n_tr_epochs)

    t_f = time.time()
    print("Training time taken: {} sec".format(t_f - t_i))

    #############################################################
    # Testing
    #############################################################

    test_set = generate_five_param(eval_dataset_size, res)
    def eval_fn():
        return (test_set
                .shuffle(eval_dataset_size*2)
                .repeat(1)
                .batch(batch_size)
                .make_one_shot_iterator()
                .get_next())
    eval_spec = tf.estimator.EvalSpec(
            input_fn = eval_fn,
            steps = int(eval_dataset_size/batch_size))

    # TODO Add more interesting metrics to check during evaluation time
    #  logging_hook = tf.train.LoggingTensorHook(
        #  tensors={"loss_c": "l2_loss"}, every_n_iter=5)

    (eval_result, export_strategy) = tf.estimator.train_and_evaluate(
            regressor, 
            train_spec, eval_spec)
    #  eval_batches = int(eval_dataset_size/batch_size)
    #  eval_result = regressor.evaluate(input_fn=eval_fn, steps=eval_batches)
    print("RMSE for test set: {}".format(eval_result["rmse"]))
    print("Relative Error for test set: {}".format(eval_result["rel_err"]))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
