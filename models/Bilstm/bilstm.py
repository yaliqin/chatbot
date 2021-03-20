from util import make_w2v_embeddings, make_w2v_embeddings_after_train, split_and_zero_padding, ManDist
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import gensim
from gensim.models import KeyedVectors
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, \
    Bidirectional, TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D
from keras.layers.merge import multiply, concatenate
import keras.backend as K

train_df = pd.read_csv("/Users/wenyaxie/Downloads/train.csv")
train_df['U_n'] = [[] for _ in range(len(train_df.index))]
train_df['R_n'] = [[] for _ in range(len(train_df.index))]

valid_df = pd.read_csv("/Users/wenyaxie/Downloads/valid.csv")
valid_df['U_n'] = [[] for _ in range(len(valid_df.index))]
valid_df['R_n'] = [[] for _ in range(len(valid_df.index))]

after_train_df = train_df.loc[train_df['flag'] == 1]  # Used for finding a match for given user input
answers = []
for index, row in after_train_df.iterrows():
    answers.append(row['R'])

embedding_path = '/Users/wenyaxie/Downloads/Chatbot_bilstm/GoogleNews-vectors-negative300'
embedding_dim = 300
embedding_dict = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
max_seq_length = 200

train_size = len(train_df)
df = train_df.append(valid_df, ignore_index=True)
df, embeddings, vocabs = make_w2v_embeddings(embedding_dict, df, embedding_dim=embedding_dim)
train_df = df[:train_size]
valid_df = df[train_size:]

X_train = train_df[['U_n', 'R_n']]
Y_train = train_df['flag']

X_validation = valid_df[['U_n', 'R_n']]
Y_validation = valid_df['flag']

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

Y_train = Y_train.values
Y_validation = Y_validation.values

assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


def shared_model(_input):
    embedded = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,),
                         trainable=False)(_input)

    activations = Bidirectional(LSTM(n_hidden, return_sequences=True), merge_mode='concat')(embedded)
    activations = Bidirectional(LSTM(n_hidden, return_sequences=True), merge_mode='concat')(activations)

    activations = Dropout(0.5)(activations)

    attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(n_hidden * 2)(attention)
    attention = Permute([2, 1])(attention)
    sent_representation = multiply([activations, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    sent_representation = Dropout(0.1)(sent_representation)

    return sent_representation


def shared_model_cnn(_input):
    embedded = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,),
                         trainable=False)(_input)

    # CNN
    activations = Conv1D(250, kernel_size=5, activation='relu')(embedded)
    activations = GlobalMaxPool1D()(activations)
    activations = Dense(250, activation='relu')(activations)
    activations = Dropout(0.3)(activations)
    activations = Dense(1, activation='sigmoid')(activations)

    return activations


if __name__ == '__main__':

    batch_size = 1024
    n_epoch = 9
    n_hidden = 50

    left_input = Input(shape=(max_seq_length,), dtype='float32')
    right_input = Input(shape=(max_seq_length,), dtype='float32')
    left_sen_representation = shared_model(left_input)
    right_sen_representation = shared_model(right_input)

    man_distance = ManDist()([left_sen_representation, right_sen_representation])
    sen_representation = concatenate([left_sen_representation, right_sen_representation, man_distance])
    similarity = Dense(1, activation='sigmoid')(Dense(2)(Dense(4)(Dense(16)(sen_representation))))
    model = Model(inputs=[left_input, right_input], outputs=[similarity])

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    training_start_time = time()
    malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                               batch_size=batch_size, epochs=n_epoch,
                               validation_data=([X_validation['left'], X_validation['right']], Y_validation))
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))

    user_input = "How can I reset my password? I can’t remember my password.   Currently we do not have this facility. Users will need to contact the admin  How can I contact the admin? Can you give me the contact information?"
    replaced_df = after_train_df.copy()
    for index, row in replaced_df.iterrows():
        replaced_df.at[index, "R"] = user_input
    replaced_df = make_w2v_embeddings_after_train(replaced_df, vocabs)
    X_replaced = replaced_df[['U_n', 'R_n']]
    X = split_and_zero_padding(X_replaced, max_seq_length)
    input_prediction = model.predict([X['left'], X['right']])
    max_score = 0
    index = 0
    for i in range(len(input_prediction)):
        score = input_prediction[i]
        if score > max_score:
            max_score = score
            index = i
    print(max_score)
    print(answers[index])
