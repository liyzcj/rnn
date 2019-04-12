from model.inputs import RAEInputs
from model.model import RAEModel
import os
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 随着进程逐渐添加显存占用

class Params:
    batch_size = 20
    time_steps = 20
    hidden_units = 200
    keep_prob = 1.0
    num_layers = 1
    grad_clip = 5
    num_epoch = 1
    learninig_rate = 1.0
    decay_epoch = 4
    lr_decay = 0.5
    embedding_size = 200
    vocab_size = 10000
params = Params()

save_path = "rnnae/save/model1"
data_path = "test/data/"
train_path = os.path.join(data_path, "ptb.train.txt")
valid_path = os.path.join(data_path, "ptb.valid.txt")
test_path = os.path.join(data_path, "ptb.test.txt")
vocab = tf.contrib.lookup.index_table_from_file("test/data/vocab.tsv")

if not os.path.exists(train_path):
    raise Exception("no such file.")

with tf.name_scope("Input"):
    inputs = RAEInputs(valid_path, params, vocab)
with tf.variable_scope("Model"):
    model = RAEModel(params, inputs)

# Initialize Operation
init = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    # Initialization
    sess.run(init)
    sess.run(inputs.init_op)
    writer = tf.summary.FileWriter(save_path, sess.graph)

    for i in range(params.num_epoch):
        lr_decay = params.lr_decay ** max(i+1-params.decay_epoch, 0.0)
        model.assign_lr(sess,params.learninig_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(model.lr)))
        total_loss = model.run_one_epoch(sess, writer)
        print("Epoch: %d Total Loss: %.3f" % (i + 1, total_loss))
        # Save the model
        path = os.path.join(save_path, 'after-epoch')
        saver.save(sess, path, global_step=i+1)
    # try:
    #     while(True):
    #         batch = sess.run(inputs.next_batch)
    #         sentence = batch['sentence']
    #         length = batch['length']
    #         print(f"Sentence: {sentence}")
    #         print(f"Length: {length}")
    # except tf.errors.OutOfRangeError:
    #     print("end")