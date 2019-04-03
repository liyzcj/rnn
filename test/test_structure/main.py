from inputs import SEMInputs
from model import SEMModel
import os
import tensorflow as tf

class Params:
    batch_size = 4
    time_steps = 5
    feature_dim = 512
    hidden_units = 200
    keep_prob = 1.0
    num_layers = 1
    grad_clip = 5
    num_epoch = 3
    learninig_rate = 1
    decay_epoch = 2
    lr_decay = 0.5

params = Params()

data_path = "test/data/"
train_path = os.path.join(data_path, "ptb.train.txt.npy")
valid_path = os.path.join(data_path, "ptb.valid.txt.npy")
test_path = os.path.join(data_path, "ptb.test.txt.npy")

if not os.path.exists(train_path):
    raise Exception("no such file.")



train_inputs = SEMInputs(train_path, params)
with tf.name_scope("Train"):
    
    with tf.variable_scope("Variables"):
        train_model = SEMModel(params, train_inputs, is_training = True)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
sv = tf.train.Supervisor(logdir="test/test_structure/logs", init_op=init)
saver=sv.saver

with sv.managed_session() as sess:    
    for epoch in range(params.num_epoch):
        lr_decay = params.lr_decay ** max(epoch + 1 - params.decay_epoch, 0.0)
        train_model.assign_lr(sess, params.learninig_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (epoch + 1, sess.run(train_model.lr)))
        train_loss = train_model.run_one_epoch(sess)
        print("Epoch: %d Total Train Loss: %.3f" % (epoch + 1, train_loss))
        globa = sess.run(train_model.global_step)
        print("Global_step = ", globa)
    #     valid_loss = one_epoch(sess, valid_model, valid_inputs, params, is_training=False)
    #     print("Epoch: %d Valid Perplexity: %.3f" % (epoch + 1, valid_loss))
    # test_loss = one_epoch(sess, test_model, test_inputs, params, is_training=False)
    # print("Test Loss : %.3f" % test_loss)