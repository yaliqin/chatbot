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
import preprocess.preprocessor as preprocessor
import utils.predict as predict

# configure
#checkpoint_path ="./output/ubuntu/DAM"
#latest = tf.train.latest_checkpoint(checkpoint_path)

data_path =  "/Users/ally/Documents/12020Fall/data298/github/chatbot/data/"
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
    # "init_meta":"./output/ubuntu/DAM/DAM.ckpt.meta",
    # "init_model":"./output/ubuntu/DAM/DAM.ckpt",
    "init_model":"", # for local machine test

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
# train.train(conf, model)

data_file = data_path+"original_data2.txt"
corpus = preprocessor.read_txt_file(data_file)
texts = preprocessor.get_texts(corpus)
word_dict = preprocessor.generate_word_dict(texts)
answers_text= []
question_text = []
positive_corpus =[]
for item in corpus:
    blocks = item.split('\t')
    if(blocks[0]=='1'):
        answers_text.append(([blocks[-1]]))
        question_text.append((blocks[1:-1]))
        positive_corpus.append(item)
question_number = [1,10]
all_positive_answers = predict.build_candidate_answers(positive_corpus, word_dict)

for item in question_number:
    print(f'the {item} question is:{question_text[item]}')
    print(f'the question of {item} question is:{answers_text[item]}')
# with open(data_file, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         blocks = line.split('\t')
#         if(blocks[0]=='1'):
#             answers_text.append([blocks[-1]])
#     print(answers_text)
# f.close()


# with open(data_file, "r") as f:
#     lines = f.readlines()
#     print('the question is:')
#     line = lines[number_n_question]
#     blocks = line.split('\t')
#     print(blocks[1:-1])
# f.close()

    question = predict.build_question(positive_corpus, item, word_dict)
    all_positive_data = predict.generate_data(question, all_positive_answers, word_dict)

    answer,index = predict.test(conf, model, all_positive_data)
    print(f'answer index is {index}')
    print(f'for the question:{question}, the answer is: \n')
    print(answer)
    print(all_positive_answers[index])
# test.test(conf, model)


