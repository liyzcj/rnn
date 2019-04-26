from inputs import PTBInputs
from model import PTBModel
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# config = tf.ConfigProto(graph_options=tf.GraphOptions(
#         optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 随着进程逐渐添加显存占用

class Params:
    batch_size = 20
    time_steps = 20
    hidden_units = 200
    keep_prob = 1.0
    num_layers = 2
    grad_clip = 5
    num_epoch = 1
    learninig_rate = 1.0
    decay_epoch = 4
    lr_decay = 0.5
    embedding_size = 200
    vocab_size = 10000

restore_path = "test/language_model/save/model1"
save_path = "test\\language_model\\save\\model1"
data_path = "test/data/"
train_path = os.path.join(data_path, "ptb.train.txt")
valid_path = os.path.join(data_path, "ptb.valid.txt")
test_path = os.path.join(data_path, "ptb.test.txt")
words = tf.contrib.lookup.index_table_from_file("test/data/vocab.tsv")

if not os.path.exists(train_path):
    raise Exception("no such file.")

train_params = Params()
test_params = Params()
test_params.batch_size = 1
test_params.time_steps = 1

initializer = tf.random_uniform_initializer(-0.1,0.1)

with tf.name_scope("Train"):
    with tf.name_scope("TrainInputs"):
        train_input = PTBInputs(train_path, train_params, words)
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        train_model = PTBModel(train_params, train_input, is_training=True)

with tf.name_scope("Valid"):
    with tf.name_scope("ValidInputs"):
        valid_input = PTBInputs(valid_path, train_params, words)
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
        valid_model = PTBModel(train_params, valid_input)
    
with tf.name_scope("Test"):
    with tf.name_scope("TestInputs"):
        test_input = PTBInputs(test_path, test_params, words)
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
        test_model = PTBModel(test_params, test_input)

init = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])

saver = tf.train.Saver()
# sv = tf.train.Supervisor(logdir="test/language_model/logs/", init_op=init)
# saver=sv.saver


# with sv.managed_session(config=config) as sess:
with tf.Session(config=config) as sess:
    sess.run(init)
    sess.run(train_input.init_op)
    sess.run(valid_input.init_op)
    sess.run(test_input.init_op)
    writer = tf.summary.FileWriter(save_path, sess.graph)

    # Projector for embedding in tensorboard
    proj_config = projector.ProjectorConfig()
    embed = proj_config.embeddings.add()
    embed.tensor_name = train_model.embedding.name
    # spicfy the meta data for projector
    embed.metadata_path = "vocab.tsv"
    projector.visualize_embeddings(writer, proj_config)

    bae = 0
    # Reload weights if exits
    if os.path.exists(restore_path):
        print("Restoring parameters from {}".format(restore_path))
        if os.path.isdir(restore_path):
            restore_path = tf.train.latest_checkpoint(restore_path)
            # Begin at epoch
        bae = int(restore_path.split('-')[-1])
        saver.restore(sess, restore_path)

    # end of epoch 
    eoe = bae + train_params.num_epoch
    for i in range(bae, eoe):
        lr_decay = train_params.lr_decay ** max(i+1-train_params.decay_epoch, 0.0)
        train_model.assign_lr(sess,train_params.learninig_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(train_model.lr)))
        train_perplexity = train_model.run_one_epoch(sess, writer)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = valid_model.run_one_epoch(sess)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        # Save model
        path = os.path.join(save_path, 'after-epoch')
        saver.save(sess, path, global_step=i+1)
    # test_perplexity = test_model.run_one_epoch(sess)
    # print("Test Perplexity: %.3f" % test_perplexity)