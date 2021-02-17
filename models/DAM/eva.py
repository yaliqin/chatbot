import sys
import os
import time

# import cPickle as pickle
import pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva
save_path = "./output/ubuntu/DAM/",

score_file_path = save_path + 'score.test'

result = eva.evaluate(score_file_path)
result_file_path = save_path + "result2.test"
with open(result_file_path, 'w') as out_file:
    for p_at in result:
        out_file.write(str(p_at) + '\n')
print('finish evaluation')