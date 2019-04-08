from inputs import PTBInputs
import tensorflow as tf
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 随着进程逐渐添加显存占用

class Params:
    batch_size = 20
    time_steps = 20
    hidden_units = 200
    keep_prob = 1.0
    num_layers = 2
    grad_clip = 5
    num_epoch = 10
    learninig_rate = 1.0
    decay_epoch = 4
    lr_decay = 0.5
    embedding_size = 200
    vocab_size = 10000
params = Params()


data_path = "test/data/"
train_path = os.path.join(data_path, "ptb.train.txt")
valid_path = os.path.join(data_path, "ptb.valid.txt")
test_path = os.path.join(data_path, "ptb.test.txt")
words = tf.contrib.lookup.index_table_from_file("test/data/vocab.txt")


valid = PTBInputs(valid_path, params, words)
batch = valid.next_batch



with tf.Session(config=config) as sess:

    sess.run(valid.init_op)
    sess.run(tf.tables_initializer())
    print(sess.run(batch))