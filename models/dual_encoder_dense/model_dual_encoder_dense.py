import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys


def dot_semantic_nn(context, utterance, tng_mode):

    keep_prob = 0.5
    if tng_mode == ModeKeys.TRAIN:
        keep_prob = 0.5


    context_channel = _network_channel(network_name='context_channel', net_input=context, keep_prob=keep_prob)


    utterance_channel = _network_channel(network_name='utterance_channel', net_input=utterance, keep_prob=keep_prob)


    mean_loss = _negative_log_probability_loss(context_channel, utterance_channel)
    K = tf.matmul(context_channel, utterance_channel, transpose_b=True)


    return mean_loss, context_channel, utterance_channel, K


def _negative_log_probability_loss(context_channel, utterance_channel):


    K = tf.matmul(context_channel, utterance_channel, transpose_b=True)


    S = tf.diag_part(K)
    S = tf.reshape(S, [-1, 1])


    K = tf.reduce_logsumexp(K, axis=1, keep_dims=True)


    return -tf.reduce_mean(S - K)


def _network_channel(network_name, net_input, keep_prob):

    with tf.variable_scope(network_name) as scope:
        predict_opt_name = '{}_branch_predict'.format(network_name)


        with tf.variable_scope('dense_branch') as d_scope:
            dense_0 = tf.layers.dense(net_input, units=300, activation=tf.nn.tanh)
            dense_0 = tf.layers.batch_normalization(dense_0)
            dense_0 = tf.layers.dropout(inputs=dense_0, rate=keep_prob)

            dense_1 = tf.layers.dense(dense_0, units=300, activation=tf.nn.tanh)
            dense_1 = tf.layers.batch_normalization(dense_1)
            dense_1 = tf.layers.dropout(inputs=dense_1, rate=keep_prob)

            dense_2 = tf.layers.dense(dense_1, units=500, activation=tf.nn.tanh, name=predict_opt_name)
            tf.add_to_collection('{}_embed_opt'.format(network_name), dense_2)

        return dense_2
