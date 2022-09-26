import os, sys

import keras.losses
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
from keras_preprocessing.sequence import pad_sequences

input_sentences = []
output_sentences = []
# output_sentences_inputs = []
for line in open('./activePassive.txt','r'):
    input_text,output_text = line.split(',')
    checkArr = input_text.split(' ')
    if(len(checkArr)>10):
        continue
    input_text = input_text.lower()
    output_text = output_text.lower()
    output_text = output_text.strip()
    checkArr = []
    for word in output_text.split(" "):
        if word in input_text:
            checkArr.append(word)
    output_text = " "
    output_text = output_text.join(checkArr)
    # output_sentence = output_text + ' <eos>'
    # output_sentences_input = '<sos> ' + output_text
    input_sentences.append(input_text)
    output_sentences.append(output_text)
    # output_sentences_inputs.append

print(input_sentences)
print(output_sentences)

input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_sentences)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)
print(input_integer_seq)
word2idx_inputs = input_tokenizer.word_index
print(word2idx_inputs)
max_input_sen_len = max(len(sentence) for sentence in input_integer_seq)
print(max_input_sen_len)

lstm_input_sequences = pad_sequences(input_integer_seq,maxlen=max_input_sen_len)
lstm_input_sequences = np.array(lstm_input_sequences)
lstm_input_sequences = np.reshape(lstm_input_sequences,(68,1,10))
print(lstm_input_sequences)

output_prob_matrix = np.zeros((len(input_sentences),max_input_sen_len,10))
for index,sentence in enumerate(input_sentences):
    output_sentence = output_sentences[index]
    output_sentence = output_sentence.split(" ")
    sentence = sentence.split(' ')
    for word_index,word in enumerate(sentence):
        for output_index,output_word in enumerate(output_sentence):
            if output_word==word:
                output_prob_matrix[index,word_index,output_index] = 1
                break

print("here:")
print(output_prob_matrix[0])

lstm = Sequential()
lstm.add(LSTM(10, return_sequences=True,  input_shape=(1, 10)))
lstm.add(Dense(1))

lstm.add(Dense(units=10, activation='sigmoid'))
loss_fn = keras.losses.MeanSquaredError()
lstm.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
lstm.summary()
lstm.fit(lstm_input_sequences, output_prob_matrix, 32, 400)





# lstm = Sequential()
# lstm.add(LSTM(10,input_shape=(1,10)))
# lstm.add(Dense(1))
# feature_vec = lstm(lstm_input_sequences)
# feature_vec = np.array(feature_vec)
# # np.reshape(feature_vec,(68,10,10))
# vnn = Sequential()
# vnn.add(Input(1,68))
# vnn.add(Dense(units=10,activation='sigmoid'))
# loss_fn = keras.losses.MeanSquaredError()
# vnn.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
# print(vnn.summary())
# vnn.fit(feature_vec,output_prob_matrix,32,100)







# output_tokenizer = Tokenizer(num_words=70, filters='')
# output_tokenizer.fit_on_texts(output_sentences+output_sentences_inputs)
# output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
# output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)
# word2idx_outputs = output_tokenizer.word_index
# num_words_output = len(word2idx_outputs) + 1
# max_output_sen_len = max(len(sentence) for sentence in output_integer_seq)
#
# encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_sen_len)
#
# decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_output_sen_len, padding='post')
#
# embeddings_dictionary = dict()
# glove_file = open('./glove.6B.100d.txt', 'r',encoding="utf8")
# for line in glove_file:
#     word_record = line.split()
#     word = word_record[0]
#     vector_dim = np.asarray(word_record[1:],dtype='float')
#     embeddings_dictionary[word] = vector_dim
# glove_file.close()
#
# num_words = len(word2idx_inputs)+1
# embedding_matrix = np.zeros((num_words,100))
# for word,i in word2idx_inputs.items():
#     embedding_vector = embeddings_dictionary.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
#
# embedding_layer = Embedding(num_words, 100, weights=[embedding_matrix], input_length=max_output_sen_len)
#
# decoder_targets_one_hot = np.zeros((len(input_sentences),max_output_sen_len,num_words_output),dtype='float32')
# for i, d in enumerate(decoder_input_sequences):
#     for t, word in enumerate(d):
#         decoder_targets_one_hot[i, t, word] = 1
#
# encoder_inputs_placeholder = Input(shape=(max_input_sen_len,))
# encoder_inputs_x = embedding_layer(encoder_inputs_placeholder)
# encoder_lstm = LSTM(256, return_state=True)
# encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs_x)
# encoder_states = [state_h, state_c]
#
# decoder_inputs_placeholder = Input(shape=(max_output_sen_len,))
# decoder_embedding = Embedding(num_words_output, 256)
# decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
# decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)
#
# decoder_dense = Dense(num_words_output, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)
#
# model = Model([encoder_inputs_placeholder,decoder_inputs_placeholder], decoder_outputs)
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(
#     [encoder_input_sequences, decoder_input_sequences],
#     decoder_targets_one_hot,
#     batch_size=64,
#     epochs=200,
#     validation_split=0.1,
# )
#
# encoder_model = Model(encoder_inputs_placeholder, encoder_states)
# decoder_state_input_h = Input(shape=(256,))
# decoder_state_input_c = Input(shape=(256,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_inputs_single = Input(shape=(1,))
# decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
# decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)
# decoder_states = [h, c]
# decoder_outputs = decoder_dense(decoder_outputs)
# decoder_model = Model(
#     [decoder_inputs_single] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states
# )
#
# idx2word_inputs = {i:w for w,i in word2idx_inputs.items()}
# idx2word_outputs = {i:w for w,i in word2idx_outputs.items()}
# def getOutput(input):
#     states = encoder_model.predict(input)
#     target_seq = np.zeros((1,1))
#     target_seq[0,0] = word2idx_outputs['<sos>']
#     eos_index = word2idx_outputs['<eos>']
#     output_sen = []
#     for _ in range(max_output_sen_len):
#         output_tokens, h, c = decoder_model.predict([target_seq] + states)
#         index = np.argmax(output_tokens[0, 0, :])
#         if eos_index == index:
#             break
#         word = ''
#         if index>0:
#             word = idx2word_outputs[index]
#             output_sen.append(word)
#         target_seq[0,0] = index
#         states = [h,c]
#     return ' '.join(output_sen)
#
# random_index = 15
# input = encoder_input_sequences[random_index:random_index+1]
# output_generated = getOutput(input)
# print(input_sentences[random_index])
# print(output_generated)
# BATCH_SIZE = 64
# EPOCHS = 20
# LSTM_NODES =256
# NUM_SENTENCES = 20000
# MAX_SENTENCE_LENGTH = 50
# EMBEDDING_SIZE = 100