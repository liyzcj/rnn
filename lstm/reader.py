
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  # 将文本中的词按照顺序存入list中
  data = _read_words(filename) 
  #以字典的形式返回序列（list，字符串）中各元素的频数
  counter = collections.Counter(data) 
  '''list1 = [3,5,-4,-1,0,-2,-6]      
  c=sorted(list1, key=lambda x: abs(x)) 按照绝对值升序排列
    key=lambda x: abs(x) #将元素值先执行绝对值运算，再进行比较'''
   #key=lambda x: (-x[1], x[0])  优先按照第一维的逆序（降序）排列，再按照第〇维的（正序）升序排列
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
 
  #将键值对转换成列表形式，所有键在前列，所有键的值在后列。words获取所有键
  words, _ = list(zip(*count_pairs))
  # 将所有的键编写一个0到len(words)的索引，并保存到字典中
  word_to_id = dict(zip(words, range(len(words))))
  #返回words和索引的字典
  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")
  #_build_vocab函数对字典对象，先按value(频数)降序，频数相同的单词再按key(单词)升序。函数返回的是字典对象， 
    # 函数返回的是字典对象，key为单词，value为对应的唯一的编号
  
  word_to_id = _build_vocab(train_path)
  # _file_to_word_ids函数，用于把文件中的内容转换为索引列表。在转换过程中，若文件中的某个单词不在word_to_id查询字典中，
  # 则不进行转换。返回list对象，list中的每一个元素均为int型数据，代表单词编号
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  #词汇表的大小
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    #print(len(raw_data))
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    
    # 这里的batch_size指某一时刻输入单词的个数。因为程序在执行时要利用GPU的并行计算能力提高效率，所以程序设定了这个参数
    batch_len = data_len // batch_size
    #将所有的词转换成batch_size乘batch_len的矩阵
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])
    # TF仅支持定长输入，这里设定RNN网络的序列长度为num_steps
    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")
    #i = tf.train.range_input_producer(NUM_EXPOCHES, num_epochs=1, shuffle=False).dequeue()
    #inputs = tf.slice(array, [i * BATCH_SIZE], [BATCH_SIZE])
    #第一行会产生一个队列，队列包含0到NUM_EXPOCHES-1的元素，
    #如果num_epochs有指定，则每个元素只产生num_epochs次，否则循环产生。
    #shuffle指定是否打乱顺序，这里shuffle=False表示队列的元素是按0到NUM_EXPOCHES-1的顺序存储。
    #在Graph运行的时候，每个线程从队列取出元素，假设值为i，然后按照第二行代码切出array的一小段数据作为一个batch。
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
     
    #在data词矩阵中获取参与第i次迭代的训练词
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    #在data词矩阵中获取参与第i次迭代的训练词对应的目标词
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
     
 
    return x, y
