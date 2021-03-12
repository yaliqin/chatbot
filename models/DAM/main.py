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
sys.path.append("../../preprocess")
import preprocessor
import generate_data
#import preprocess.preprocessor as preprocessor
import utils.predict as predict
#import utils.generate_data as generate_data

# configure
#checkpoint_path ="./output/ubuntu/DAM"
#latest = tf.train.latest_checkpoint(checkpoint_path)

data_path =  "/home/ally/github/chatbot/data/"
result_path = "/home/ally/github/chatbot/models/DAM/results/"
conf = {
    #"data_path": "./data/ubuntu/data.pkl",
    "data_path": data_path+"classified_split.pickle",
    "train_path":data_path+"classified_train.pickle",
    "valid_path":data_path+"classified_valid.pickle",
    "test_path":data_path+"classified_test.pickle",
    #"save_path": "./output/ubuntu/temp/",
    # "word_emb_init": "./data/word_embedding.pkl",
    "word_emb_init":None,

    "save_path": result_path,
    #"init_model":"/home/ally/DAM/output/ubuntu/DAM/DAM.ckpt",
    #"init_meta":"/home/ally/DAM/output/ubuntu/DAM/DAM.ckpt.meta",
    "init_model":result_path+"model.ckpt.21.0", # for local machine test
    "init_meta":result_path+"model.ckpt.21.0.meta",

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
    "batch_size": 10, #200 for test

    "max_turn_num": 9,  
    "max_turn_len": 50, 

    "max_to_keep": 1,
    "num_scan_data": 2,
    "_EOS_": 28270, #1 for douban data
    "final_n_class": 1,
}




model = net.Net(conf)
#print(conf)
#train.train(conf, model)

data_file = data_path+"all_classified_data.txt"
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

cls_indexs, question_text, answers_text =  generate_data.get_subset_answers(data_path)
print(cls_indexs)

key_words_list = ["input classification", "output", "context"]

question_number = [120]
#all_positive_answers = predict.build_candidate_answers(positive_corpus, word_dict)
#
for index in question_number:
    print(f'the {index} question is:{question_text[index]}')
    question = question_text[index]
    positive_answer, negative_answers,negative_index_list = generate_data.generate_candidate_answers(question, key_words_list, cls_indexs, question_text, answers_text)
    all_data = []
    all_data.append(positive_answer[0])
    print(positive_answer[0])
    for item in negative_answers:
      all_data.append(item)
      print(item)
#      print("\n")
#    print(f'all_data is {all_data}')
    print("*********************************************\n")

#    #print(f'the question of {item} question is:{answers_text[item]}')
#
#    texts = preprocessor.get_texts(all_data)
    text_data_classified = preprocessor.get_sequence_tokens_with_turn(all_data,word_dict)
    print(text_data_classified)
#    question = predict.build_question(positive_corpus, item, word_dict)
#    all_positive_data = predict.generate_data(question, all_positive_answers, word_dict)
#
    answer,answer_index = predict.test(conf, model, text_data_classified)
    negative_index_list.insert(0,index)
    print(f'candiate answers index:{negative_index_list}')
    print(f'answer index is {index} in the classification list')
    print(f'for the question:{question}, the answer is: \n')
    print(answer)

    print(answers_text[negative_index_list[answer_index]])
# test.test(conf, model)


