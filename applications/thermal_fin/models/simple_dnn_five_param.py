import tensorflow as tf

def simple_dnn(features, labels, mode, params):
    '''
    A simple deep neural network to learn the ROM error
    for parametric system of 5 real values.

    Arguments:
        features - This is batch_features from input_fn
        labels   - This is batch_labels from input_fn
        mode     - An instance of tf.estimator.ModeKeys, see below
        params   - Additional configuration
    '''

    batch_size = params["batch_size"]
    activation_fn = tf.nn.sigmoid
    layer_width = 80
    #  activation_fn = tf.nn.relu

    # input_layer shape = [batch_size, num_nodes]
    # -1 for batch size specifies that this dimension should be dynamically
    dense1 = tf.layers.dense(features, units=layer_width, activation=activation_fn)

    dense2 = tf.layers.dense(dense1, units=layer_width, activation=activation_fn)

    #  dense3 = tf.layers.dense(dense2, units=ceil(num_nodes/2), activation=activation_fn)
    dense3 = tf.layers.dense(dense2, units=layer_width, activation=activation_fn)
    dense4 = tf.layers.dense(dense3, units=layer_width, activation=activation_fn)
    dense5 = tf.layers.dense(dense4, units=layer_width, activation=activation_fn)
    dense6 = tf.layers.dense(dense5, units=layer_width, activation=activation_fn)
    dense7 = tf.layers.dense(dense6, units=layer_width, activation=activation_fn)

    e_pred = tf.layers.dense(inputs=dense7, units=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=e_pred)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, e_pred)
    #  loss = tf.losses.mean_squared_error(labels, e_pred, tf.reciprocal(tf.abs(labels)))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params["optimizer"](learning_rate = params["learning_rate"])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #TODO: Add variance of loss?

    # Calculate root mean squared error
    rmse = tf.metrics.root_mean_squared_error(labels, e_pred)

    # Calculates relative error normalized by the real error
    rel_err = tf.metrics.mean_relative_error(labels, e_pred, tf.abs(labels))

    tf.summary.scalar('relative_error', rel_err)
    tf.summary.scalar('rmse', rmse)

    eval_metric_ops = {"rmse":rmse, "rel_err":rel_err}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
