"""
CasCN

"""

import tensorflow as tf
import numpy as np
import collections
import scipy.sparse
from scipy.sparse import coo_matrix
from tensorflow.contrib.layers.python.layers import regularizers

# Test for tf1.0

tfversion_ = tf.VERSION.split(".")
global tfversion
if int(tfversion_[0]) < 1:
    raise EnvironmentError("TF version should be above 1.0!!")
if int(tfversion_[1]) < 1:
    print("Working in TF version 1.0....")
    from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

    tfversion = "old"
else:
    print("Working in TF version 1.%d...." % int(tfversion_[1]))
    from tensorflow.python.ops.rnn_cell_impl import RNNCell

    tfversion = "new"


def cheby_conv(x, L, lmax, batch_size, num_nodes, feat_out, K, W):
    nSample = batch_size  # 32
    nNode = num_nodes
    x = tf.reshape(x, (nSample, num_nodes, -1))
    feat_in = x.get_shape()[2].value
    x_l = []
    # Transform to Chebyshev basis
    L1 = tf.unstack(L, nSample, 0)
    X0 = tf.unstack(x, nSample, 0)

    def concat(x, x_):
        _x = tf.expand_dims(x_, 0)
        return tf.concat([x, _x], axis=0)

    for l, x0 in zip(L1, X0):
        x_ = tf.expand_dims(x0, 0)
        if K > 1:
            x1 = tf.matmul(l, x0)
            x_ = concat(x_, x1)
        for k in range(2, K):
            x2 = 2 * tf.matmul(l, x1) - x0
            x_ = concat(x_, x2)
            x0, x1 = x1, x2
        x_l.append(x_)
    x_l = tf.reshape(x_l, [K, nNode, feat_in, nSample])
    x_l = tf.transpose(x_l, perm=[3, 1, 2, 0])
    x_l = tf.reshape(x_l, [nSample * nNode, feat_in * K])
    x_l = tf.matmul(x_l, W)  # No Bias term?? -> Yes[batch size*nNode,feat_out] [32*200,32]
    out = tf.reshape(x_l, [nSample, nNode, feat_out])
    return out


# gconvLSTM
_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ('c', 'h'))


class LSTMStateTuple(_LSTMStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state")
        return c.dtype


class gcnRNNCell(RNNCell):
    def __init__(self, num_units, forget_bias=1.0, batch_size=None,
                 state_is_tuple=True, activation=None, reuse=None,
                 laplacian=None, lmax=None, K=None, feat_in=None, nNode=None):
        if tfversion == 'new':
            super(gcnRNNCell, self).__init__(_reuse=reuse)

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.tanh
        self._laplacian = laplacian
        self._lmax = lmax
        self._K = K
        self._feat_in = feat_in
        self._nNode = nNode
        self._batch_size = batch_size

    @property
    def state_size(self):
        return (LSTMStateTuple((self._nNode, self._num_units), (self._nNode, self._num_units))
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "myZeroState"):
            zero_state_c = tf.zeros([self._batch_size, self._nNode, self._num_units], name='c')
            zero_state_h = tf.zeros([self._batch_size, self._nNode, self._num_units], name='h')
            return (zero_state_c, zero_state_h)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

            laplacian = self._laplacian
            lmax = self._lmax
            K = self._K
            feat_in = self._feat_in
            nNode = self._nNode
            batch_size = self._batch_size

            if feat_in is None:
                # Take out the shape of input
                batch_size, nNode, feat_in = inputs.get_shape()

            feat_out = self._num_units

            if K is None:
                K = 2

            scope = tf.get_variable_scope()
            with tf.variable_scope(scope) as scope:
                try:
                    # Need four diff Wconv weight + for Hidden weight
                    Wzxt = tf.get_variable("Wzxt", [K * feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wixt = tf.get_variable("Wixt", [K * feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wfxt = tf.get_variable("Wfxt", [K * feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Woxt = tf.get_variable("Woxt", [K * feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                    Wzht = tf.get_variable("Wzht", [K * feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wiht = tf.get_variable("Wiht", [K * feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wfht = tf.get_variable("Wfht", [K * feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Woht = tf.get_variable("Woht", [K * feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                except ValueError:
                    scope.reuse_variables()
                    Wzxt = tf.get_variable("Wzxt", [K * feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wixt = tf.get_variable("Wixt", [K * feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wfxt = tf.get_variable("Wfxt", [K * feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Woxt = tf.get_variable("Woxt", [K * feat_in, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                    Wzht = tf.get_variable("Wzht", [K * feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wiht = tf.get_variable("Wiht", [K * feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wfht = tf.get_variable("Wfht", [K * feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Woht = tf.get_variable("Woht", [K * feat_out, feat_out], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                bzt = tf.get_variable("bzt", [feat_out])
                bit = tf.get_variable("bit", [feat_out])
                bft = tf.get_variable("bft", [feat_out])
                bot = tf.get_variable("bot", [feat_out])

                # gconv Calculation
                zxt = cheby_conv(inputs, laplacian, lmax, batch_size, nNode, feat_out, K, Wzxt)

                zht = cheby_conv(h, laplacian, lmax, batch_size, nNode, feat_out, K, Wzht)
                zt = zxt + zht + bzt
                zt = tf.tanh(zt)

                ixt = cheby_conv(inputs, laplacian, lmax, batch_size, nNode, feat_out, K, Wixt)
                iht = cheby_conv(h, laplacian, lmax, batch_size, nNode, feat_out, K, Wiht)
                it = ixt + iht + bit
                it = tf.sigmoid(it)

                fxt = cheby_conv(inputs, laplacian, lmax, batch_size, nNode, feat_out, K, Wfxt)
                fht = cheby_conv(h, laplacian, lmax, batch_size, nNode, feat_out, K, Wfht)
                ft = fxt + fht + bft
                ft = tf.sigmoid(ft)

                oxt = cheby_conv(inputs, laplacian, lmax, batch_size, nNode, feat_out, K, Woxt)
                oht = cheby_conv(h, laplacian, lmax, batch_size, nNode, feat_out, K, Woht)
                ot = oxt + oht + bot
                ot = tf.sigmoid(ot)

                # c
                new_c = ft * c + it * zt

                # h
                new_h = ot * tf.tanh(new_c)

                if self._state_is_tuple:
                    new_state = LSTMStateTuple(new_c, new_h)
                else:
                    new_state = tf.concat([new_c, new_h], 1)
                return new_h, new_state


class Model(object):
    """
    Defined:
        Placeholder
        Model architecture
        Train / Test function
    """

    def __init__(self, config, n_node, sess):
        self.batch_size = config.batch_size  # bach size
        self.feat_in = config.feat_in  # number of feature
        self.feat_out = config.feat_out  # number of output feature
        self.num_nodes = config.num_nodes  # each sampel has num_nodes
        self.lmax = config.lmax
        self.sess = sess
        if config.activation == "tanh":
            self.activation = tf.tanh
        else:
            self.activation = tf.nn.relu
        self.max_grad_norm = config.max_grad_norm
        self.num_hidden = config.num_hidden  # rnn hidden layer
        self.num_kernel = config.num_kernel  # chebshevy K
        self.learning_rate = config.learning_rate
        self.n_time_interval = config.n_time_interval
        self.n_steps = config.n_steps  # number of steps
        self.n_hidden_dense1 = config.n_hidden_dense1
        self.n_hidden_dense2 = config.n_hidden_dense2
        self.scale1 = config.l1
        self.scale2 = config.l2
        self.scale = config.l1l2
        self.n_nodes = n_node
        self.initializer = tf.random_normal_initializer(stddev=config.stddev)
        self.initializer2 = tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        self.regularizer = regularizers.l1_l2_regularizer(self.scale1, self.scale2)
        self.regularizer_1 = regularizers.l1_regularizer(self.scale1)
        self.regularizer_2 = regularizers.l2_regularizer(self.scale2)
        self.model_step = tf.Variable(0, name='model_step', trainable=False)
        self._build_placeholders()
        self._build_var()
        self.pred = self._build_model()
        truth = self.y  # [32,1]

        # # Define loss and optimizer
        cost = tf.reduce_mean(tf.pow(self.pred - truth, 2)) + self.scale * tf.add_n(
            [self.regularizer(var) for var in tf.trainable_variables()])

        error = tf.reduce_mean(tf.pow(self.pred - truth, 2))
        tf.summary.scalar("error", error)

        var_list = tf.trainable_variables()

        opt1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads = tf.gradients(cost, var_list)
        grads_c = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads]  

        train_op = opt1.apply_gradients(zip(grads_c, var_list), global_step=self.model_step, name='train_op')

        self.loss = cost
        self.error = error

        self.train_op = train_op
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _build_placeholders(self):


        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_steps, self.num_nodes, self.num_nodes],
                                name="x")

        self.laplacian = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_nodes, self.num_nodes],
                                        name="laplacian")

        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name="y")
        self.time_interval_index = tf.placeholder(tf.float32,
                                                  shape=[self.batch_size, self.n_steps, self.n_time_interval],
                                                  name="time")
        self.rnn_index = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_steps],
                                        name="rnn_index")

    def _build_var(self, reuse=None):
        with tf.variable_scope('dense'):
            self.weights = {
                'dense1': tf.get_variable('dense1_weight', initializer=self.initializer([self.num_hidden,
                                                                                         self.n_hidden_dense1])),
                'dense2': tf.get_variable('dense2_weight', initializer=self.initializer([self.n_hidden_dense1,
                                                                                         self.n_hidden_dense2])),
                'out': tf.get_variable('out_weight', initializer=self.initializer([self.n_hidden_dense2, 1]))
            }
            self.biases = {
                'dense1': tf.get_variable('dense1_bias', initializer=self.initializer([self.n_hidden_dense1])),
                'dense2': tf.get_variable('dense2_bias', initializer=self.initializer([self.n_hidden_dense2])),
                'out': tf.get_variable('out_bias', initializer=self.initializer([1]))
            }
        with tf.variable_scope('time_decay'):
            self.time_weight = tf.get_variable('time_weight', initializer=self.initializer([self.n_time_interval]),
                                               dtype=tf.float32)

    def _build_model(self, reuse=None):

        with tf.variable_scope('gconv_model', reuse=reuse) as sc:
            cell = gcnRNNCell(num_units=self.num_hidden, forget_bias=1.0,
                              laplacian=self.laplacian, lmax=self.lmax,
                              feat_in=self.feat_in, K=self.num_kernel,
                              nNode=self.num_nodes, batch_size=self.batch_size)

            x_vector = tf.unstack(self.x, self.n_steps, 1)

            outputs, states = tf.contrib.rnn.static_rnn(
                cell,
                x_vector,
                dtype=tf.float32,
            )

            hidden_states = tf.transpose(tf.stack(outputs), [1, 0, 2, 3])

            hidden_states = tf.reduce_sum(hidden_states, axis=2)

            rnn_index = tf.reshape(self.rnn_index, [-1, 1])

            hidden_states = tf.reshape(hidden_states, [-1, self.num_hidden])
            hidden_states = tf.multiply(rnn_index, hidden_states)

        with tf.variable_scope('time_decay'):

            time_weight = tf.reshape(self.time_weight, [-1, 1])

            time_interval_index = tf.reshape(self.time_interval_index, [-1, 6])

            time_weight = tf.matmul(time_interval_index, time_weight)

            hidden_states = tf.multiply(time_weight, hidden_states)

            hidden_states = tf.reshape(hidden_states, [-1, self.n_steps, self.num_hidden])

            hidden_graph = tf.reduce_sum(hidden_states, reduction_indices=[1])

            self.hidden_graph = hidden_graph

        with tf.variable_scope('dense'):
            dense1 = self.activation(tf.add(tf.matmul(hidden_graph, self.weights['dense1']), self.biases['dense1']))
            dense2 = self.activation(tf.add(tf.matmul(dense1, self.weights['dense2']), self.biases['dense2']))
            pred = self.activation(tf.add(tf.matmul(dense2, self.weights['out']), self.biases['out']))
        return pred

    def train_batch(self, x, L, y, time_interval_index, rnn_index):
        _, time_weight = self.sess.run([self.train_op, self.time_weight],
                                       feed_dict={
                                           self.x: x,
                                           self.laplacian: L,
                                           self.y: y,
                                           self.time_interval_index: time_interval_index,
                                           self.rnn_index: rnn_index})
        return time_weight

    def get_error(self, x, L, y, time_interval_index, rnn_index):
        return self.sess.run(self.error, feed_dict={
            self.x: x,
            self.laplacian: L,
            self.y: y,
            self.time_interval_index: time_interval_index,
            self.rnn_index: rnn_index})

    def predict(self, x, L, y, time_interval_index, rnn_index):
        return self.sess.run(self.pred, feed_dict={
            self.x: x,
            self.laplacian: L,
            self.y: y,
            self.time_interval_index: time_interval_index,
            self.rnn_index: rnn_index})
