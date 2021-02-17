import sys
import os
import time

# import cPickle as pickle
import pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import models.net as net
import utils.evaluation as eva
#for douban
#import utils.douban_evaluation as eva

import bin.train_and_evaluate as train
import bin.test_and_evaluate as test

# configure
#_save_path = _model.saver.save(sess, conf["save_path"] + "model.ckpt." + str(step / conf["save_step"]))
checkpoint_path ="./output/ubuntu/DAM"
latest = tf.train.latest_checkpoint(checkpoint_path)

conf = {
    "data_path": "./data/ubuntu/data_small.pkl",
    "save_path": "./output/ubuntu/temp/",
    "word_emb_init": "./data/word_embedding.pkl",
    #"init_model": None, #should be set for test
    "init_model":latest,
#    "init_model":"./output/ubuntu/DAM",
    "rand_seed": None, 

    "drop_dense": None,
    "drop_attention": None,

    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,  

    "stack_num": 5,  
    "attention_type": "dot",

    "learning_rate": 1e-3,
    "vocab_size": 434512,
    "emb_size": 200,
    "batch_size": 32, #200 for test

    "max_turn_num": 9,  
    "max_turn_len": 50, 

    "max_to_keep": 1,
    "num_scan_data": 2,
    "_EOS_": 28270, #1 for douban data
    "final_n_class": 1,
}


model = net.Net(conf)
#train.train(conf, model)

# load the trained model
save_path = "DAM"
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_path)
    print("Model restored.")

#
# checkpoint.restore(save_path)
#
#test and evaluation, init_model in conf should be set

test.test(conf, model)

