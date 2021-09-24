import numpy as np
import math
import tensorflow as tf
from model.model_sparse_graph_signal import Model
import six.moves.cPickle as pickle

tf.set_random_seed(0)
import time
from model import config as cf

# DATA_PATH = "data"

n_steps = 100
tf.flags.DEFINE_integer("n_steps", n_steps, "num of step.")
tf.flags.DEFINE_integer("time_interval", cf.time_interval, "the time interval")
tf.flags.DEFINE_integer("n_time_interval", cf.n_time_interval, "the number of  time interval")
tf.flags.DEFINE_integer("num_rnn_layers", 2, "number of rnn layers .")
tf.flags.DEFINE_integer("cl_decay_steps", 1000, "cl_decay_steps .")
tf.flags.DEFINE_integer("num_kernel", 2, "chebyshev .")
tf.flags.DEFINE_float("learning_rate", 0.005, "learning_rate.")
tf.flags.DEFINE_integer("batch_size", 32, "batch size.")
tf.flags.DEFINE_integer("num_hidden", 32, "hidden rnn size.")
tf.flags.DEFINE_float("l1", 5e-5, "l1.")
tf.flags.DEFINE_float("l2", 1e-3, "l2.")
tf.flags.DEFINE_float("l1l2", 1.0, "l1l2.")
tf.flags.DEFINE_string("activation", "relu", "activation function.")
tf.flags.DEFINE_integer("training_iters", 200 * 3200 + 1, "max training iters.")
tf.flags.DEFINE_integer("display_step", 100, "display step.")
tf.flags.DEFINE_integer("n_hidden_dense1", 32, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2", 16, "dense2 size.")
tf.flags.DEFINE_string("version", "v1", "data version.")
tf.flags.DEFINE_integer("max_grad_norm", 5, "gradient clip.")
tf.flags.DEFINE_float("stddev", 0.01, "initialization stddev.")
tf.flags.DEFINE_integer("feat_in", 100, "num of feature in")
tf.flags.DEFINE_integer("feat_out", 50, "num of feature out")
tf.flags.DEFINE_integer("lmax", 2, "max L")
tf.flags.DEFINE_integer("num_nodes", 100, "number of max nodes in cascade")
config = tf.flags.FLAGS

print("l2", config.l2)
print("learning rate:", config.learning_rate)



def get_batch(x, L, y, sz, time, n_time_interval, step, batch_size, num_step):
    batch_y = np.zeros(shape=(batch_size, 1))
    batch_x = []
    batch_L = []
    batch_time_interval_index = []
    batch_rnn_index = []
    start = step * batch_size % len(x)
    for i in range(batch_size):
        id = (i + start) % len(x)
        batch_y[i, 0] = y[id]
        batch_L.append(L[id].todense())
        temp_x = []
        for m in range(len(x[id])):
            temp_x.append(x[id][m].todense())
        batch_x.append(temp_x)
        batch_time_interval_index_sample = []

        for j in range(sz[id]):
            temp_time = np.zeros(shape=(n_time_interval))
            k = int(math.floor(time[id][j] / config.time_interval))
            temp_time[k] = 1
            batch_time_interval_index_sample.append(temp_time)
        if len(batch_time_interval_index_sample) < num_step:
            for i in range(num_step - len(batch_time_interval_index_sample)):
                temp_time_padding = np.zeros(shape=(n_time_interval))
                batch_time_interval_index_sample.append(temp_time_padding)
                i = i + 1
        batch_time_interval_index.append(batch_time_interval_index_sample)
        rnn_index_temp = np.zeros(shape=(config.n_steps))
        rnn_index_temp[:sz[id]] = 1
        batch_rnn_index.append(rnn_index_temp)

    return batch_x, batch_L, batch_y, batch_time_interval_index, batch_rnn_index


version = config.version
id_train, x_train, L, y_train, sz_train, time_train, vocabulary_size = pickle.load(
    open(cf.train_pkl, 'rb'))
id_test, x_test, L_test, y_test, sz_test, time_test, _ = pickle.load(
    open(cf.test_pkl, 'rb'))
id_val, x_val, L_val, y_val, sz_val, time_val, _ = pickle.load(open(cf.val_pkl, 'rb'))

training_iters = config.training_iters
batch_size = config.batch_size
display_step = min(config.display_step, len(sz_train) / batch_size)
print("-----------------display step-------------------")
print("display step" + str(display_step))

# determine the way floating point numbers,arrays and other numpy object are displayed
np.set_printoptions(precision=2)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start = time.time()
is_training = False
model = Model(config, config.num_nodes, sess)
sess.graph.finalize()
step = 0
best_val_loss = 1000
best_test_loss = 1000
train_writer = tf.summary.FileWriter("./train", sess.graph)

# Keep training until reach max iterations or max_try
train_loss = []
max_try = 10
patience = max_try
while step * batch_size < training_iters:
    batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index = get_batch(
                                                                                x_train,
                                                                                L,
                                                                                y_train,
                                                                                sz_train,
                                                                                time_train,
                                                                                config.n_time_interval,
                                                                                step,
                                                                                batch_size,
                                                                                n_steps)
    time_decay = model.train_batch(batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index)
    train_loss.append(
        model.get_error(batch_x, batch_L, batch_y, batch_time_interval,
                        batch_rnn_index))
    if step % display_step == 0:
        #print(time_decay)
        val_loss = []
        for val_step in range(int(len(y_val) / batch_size)):
            val_x, val_L, val_y, val_time, val_rnn_index = get_batch(
                                                                     x_val,
                                                                     L_val,
                                                                     y_val,
                                                                     sz_val,
                                                                     time_val,
                                                                     config.n_time_interval,
                                                                     val_step,
                                                                     batch_size,
                                                                     n_steps)
            val_loss.append(
                model.get_error(val_x, val_L, val_y, val_time, val_rnn_index))
        test_loss = []
        for test_step in range(int(len(y_test) / batch_size)):
            test_x, test_L, test_y, test_time, test_rnn_index = get_batch(
                                                                          x_test,
                                                                          L_test,
                                                                          y_test,
                                                                          sz_test,
                                                                          time_test,
                                                                          config.n_time_interval,
                                                                          test_step,
                                                                          batch_size,
                                                                          n_steps)
            test_loss.append(
                model.get_error(test_x, test_L, test_y, test_time, test_rnn_index))

        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            best_test_loss = np.mean(test_loss)
            patience = max_try
        predict_result = []
        test_loss = []
        for test_step in range(int(len(y_test) / batch_size + 1)):
            test_x, test_L, test_y, test_time, test_rnn_index = get_batch(
                x_test,
                L_test,
                y_test,
                sz_test,
                time_test,
                config.n_time_interval,
                test_step,
                batch_size,
                n_steps)
            predict_result.extend(
                model.predict(test_x, test_L, test_y, test_time, test_rnn_index))
            test_loss.append(
                model.get_error(test_x, test_L, test_y, test_time, test_rnn_index))
        print("last test error:", np.mean(test_loss))
        pickle.dump((predict_result, y_test, test_loss), open(
            "prediction_result_" + str(config.learning_rate) + "_CasCN", 'wb'))
        print("#" + str(step / display_step) +
              ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) +
              ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) +
              ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) +
              ", Best Valid Loss= " + "{:.6f}".format(best_val_loss) +
              ", Best Test Loss= " + "{:.6f}".format(best_test_loss)
              )
        train_loss = []
        patience -= 1
        if not patience:
            break
    step += 1

print(len(predict_result), len(y_test))
print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time() - start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", best_test_loss)
