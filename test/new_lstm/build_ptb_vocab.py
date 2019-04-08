# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:19:04 2019

@author: liyz
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


"""Build vocabulary fron ptb train word"""

import collections
import sys

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with open(filename, "r") as f:
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
  # add <pad> 
  
  # words = ['<pad>'] + list(words)
  word_to_id = dict(zip(words, range(len(words)))) 
  return word_to_id

def save_vocab_to_file(word2id, filepath):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(filepath, "w") as f:
        f.write(str(word2id))


if __name__ == "__main__":
    
    file = "test/data/ptb.train.txt"
    word2id = _build_vocab(file)
    save_vocab_to_file(word2id, "test/new_lstm/vocab.txt")
    print("successful build vocabulary to 'vocab.txt' ")
    