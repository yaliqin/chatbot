import preprocess.preprocessor as preprocessor
import os
import time
import tensorflow as tf
import utils.reader
import models.net as net

def build_candidate_answers(corpus, word_dict):
    all_data_token = preprocessor.get_sequence_tokens_with_turn(corpus,word_dict)
    all_positive_answers=[]
    data_length = len(all_data_token['y'])
    for index in range(data_length):
        if all_data_token['y'][index]== 1:
            all_positive_answers.append(all_data_token['r'][index])
    return all_positive_answers

def build_question(corpus, index, word_dict):
    all_data_token = preprocessor.get_sequence_tokens_with_turn(corpus,word_dict)
    data_length = len(all_data_token['y'])
    if index >= data_length or index < 0:
        print("Index is out of the range")
        return
    else:
        question_tokens = preprocessor.get_sequence_tokens_with_turn(corpus,word_dict)
        question = question_tokens['c'][index]
        return question


def generate_data(question, positive_answers, word_dict):
    data_size = len(positive_answers)
    all_positive_data = {'c':[],'r':[]}
    if data_size == 0:
        return
    else:
        question_token = preprocessor.get_sequence_tokens_with_turn(question, word_dict)
        all_positive_data['r'].append(positive_answers)
        questions = data_size * question_token
        all_positive_data['c'].append(questions)
    return all_positive_data

def evaluate_result(data):
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    proposed_answer = sort_data[0][1]
    return proposed_answer

def test(conf, _model, predict_data):
    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print('starting loading predict data')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(predict_data['c'][:10], predict_data['r'][:10])
    print('finish loading data')

    test_batches = reader.build_batches(predict_data, conf)

    print("finish building test batches")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # refine conf
    test_batch_num = len(test_batches["response"])

    print('configurations: %s' % conf)

    _graph = _model.build_graph()
    print('build graph sucess')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    with tf.compat.v1.Session(graph=_graph) as sess:
        # _model.init.run();
        # _model.saver = tf.train.import_meta_graph("init_meta")
        _model.saver = tf.compat.v1.train.import_meta_graph(conf["init_meta"])
        print(_model.saver)
        _model.saver.restore(sess, conf["init_model"])
        print("sucess init %s" % conf["init_model"])

        batch_index = 0
        step = 0

        score_file_path = conf['save_path'] + 'score_predict.test'
        score_file = open(score_file_path, 'w')

        print('starting test')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        for batch_index in range(test_batch_num):
            print(f"batch index is: {batch_index}")
            feed = {
                _model.turns: test_batches["turns"][batch_index],
                _model.tt_turns_len: test_batches["tt_turns_len"][batch_index],
                _model.every_turn_len: test_batches["every_turn_len"][batch_index],
                _model.response: test_batches["response"][batch_index],
                _model.response_len: test_batches["response_len"][batch_index],
                _model.label: test_batches["label"][batch_index]
            }

            scores = sess.run(_model.logits, feed_dict=feed)

            for i in range(conf["batch_size"]):
                score_file.write(str(scores[i]) + '\t' +
                    str(test_batches["r"][batch_index][i]) + '\n')

        score_file.close()
        print('finish test')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        with open(score_file, 'r') as infile:
            score_data = []
            for line in infile:
                tokens = line.strip().split('\t')
                score_data.append((float(tokens[0]), int(tokens[1])))

        # write evaluation result
        result = evaluate_result(score_data)
        return result


# if __name__ == '__main__':
    # data_file = "../data/original_data2.txt"
    # corpus = preprocessor.read_txt_file(data_file)
    # texts = preprocessor.get_texts(corpus)
    # word_dict = preprocessor.generate_word_dict(texts)
    #
    # all_positive_answers = build_candidate_answers(corpus, word_dict)
    # question = "How can I reset my password? I can’t remember my password. 	 Currently we do not have this facility. Users will need to contact the admin	 How can I contact the admin? Can you give me the contact information?"
    # all_positive_data = generate_data(question,all_positive_answers,word_dict)
    #
    # model = net.Net(conf)
    # print(conf)
    # answer = test(conf, model, all_positive_data)
    # print(f'for the question:{question}, the answer is: \n')
    # print(answer)

