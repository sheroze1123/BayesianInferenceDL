import tensorflow as tf
from math import ceil

def simple_dnn(features, labels, mode, params):
    '''
    A simple deep neural network performing dropout
    in the hidden layers.

    Arguments:
        features - This is batch_features from input_fn
        labels   - This is batch_labels from input_fn
        mode     - An instance of tf.estimator.ModeKeys, see below
        params   - Additional configuration
    '''

    batch_size = tf.flags.FLAGS.batch_size
    num_nodes = params["num_nodes"]

    # input_layer shape = [batch_size, num_nodes]
    # -1 for batch size specifies that this dimension should be dynamically
    dense1 = tf.layers.dense(features, units=num_nodes, activation=tf.nn.relu)

    dense2 = tf.layers.dense(dense1, units=num_nodes, activation=tf.nn.relu)

    #  dense3 = tf.layers.dense(dense2, units=ceil(num_nodes/2), activation=tf.nn.relu)
    dense3 = tf.layers.dense(dense2, units=num_nodes, activation=tf.nn.relu)

    #  dense4 = tf.layers.dense(dense3, units=ceil(num_nodes/4), activation=tf.nn.relu)
    dense4 = tf.layers.dense(dense3, units=num_nodes, activation=tf.nn.relu)

    dense5 = tf.layers.dense(dense4, units=num_nodes, activation=tf.nn.relu)
    dense6 = tf.layers.dense(dense5, units=num_nodes, activation=tf.nn.relu)
    dense7 = tf.layers.dense(dense6, units=num_nodes, activation=tf.nn.relu)
    dense8 = tf.layers.dense(dense7, units=num_nodes, activation=tf.nn.relu)
    dense9 = tf.layers.dense(dense8, units=num_nodes, activation=tf.nn.relu)
    #  dense5 = tf.layers.dense(dense4, units=ceil(num_nodes/8), activation=tf.nn.relu)

    logits = tf.layers.dense(inputs=dense9, units=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #TODO: Add variance of loss?
    # Add evaluation metrics (for EVAL mode)
    #  eval_metric_ops = {
        #  "accuracy": tf.metrics.accuracy(
        #  labels=labels, predictions=logits)}
    eval_metric_ops = {}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
