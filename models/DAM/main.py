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
#checkpoint_path ="./output/ubuntu/DAM"
#latest = tf.train.latest_checkpoint(checkpoint_path)

data_path =  "../../data/"
conf = {
    #"data_path": "./data/ubuntu/data.pkl",
    "data_path": data_path+"data_split.pickle",
    "train_path":data_path+"train.pickle",
    "valid_path":data_path+"valid.pickle",
    "test_path":data_path+"test.pickle",
    #"save_path": "./output/ubuntu/temp/",
    # "word_emb_init": "./data/word_embedding.pkl",
    "word_emb_init":None,

    "save_path": data_path,
    #:    "init_model": "./output/ubuntu/DAM/DAM.ckpt.data-00000-of-00001", #should be set for test
    "init_meta":"/home/ally/DAM/output/ubuntu/DAM/DAM.ckpt.meta",
    "init_model":"/home/ally/DAM/output/ubuntu/DAM/DAM.ckpt",
    "init_model":data_path + "model.ckpt.15.0", # get the best model from train

    "rand_seed": None, 

    "drop_dense": None,
    "drop_attention": None,

    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,  

    "stack_num": 5,  
    "attention_type": "dot",

    "learning_rate": 1e-3,
    "vocab_size": 9449,
    "emb_size": 200,
    "batch_size": 20, #200 for test

    "max_turn_num": 9,  
    "max_turn_len": 50, 

    "max_to_keep": 1,
    "num_scan_data": 2,
    "_EOS_": 28270, #1 for douban data
    "final_n_class": 1,
}




model = net.Net(conf)
print(conf)
#train.train(conf, model)

test.test(conf, model)


