# coding=utf-8
from lstmmt import decoder
from lstmmt import data_utils
import pickle
import tensorflow as tf

with open("w2v_dict.pickle",'rb') as f:
    w2v_dict=pickle.load(f)
print ("len(w2v_dict):",len(w2v_dict))
print ("w2v_dict:",w2v_dict)

with open('embeddings.pickle','rb') as f:
            embeddings=pickle.load(f)
print ("len(embeddings):",len(embeddings))
print("embeddings:",embeddings)
W=tf.Variable(embeddings,name="W")
print ("W:",W)

# 获取数据
encoder_path = r'train.enc'
decoder_path=r'train.dec'
X_train, y_train, encoder_dict_words, encoder_index_of_words,decoder_dict_words, decoder_index_of_words = data_utils.get_vector(encoder_path,decoder_path)
encoder_dict_size = len(encoder_dict_words)
decoder_dict_size = len(decoder_dict_words)
print("encoder_dict_size:",encoder_dict_size)
print("decoder_dict_size:",decoder_dict_size)
lstm = decoder.myLSTM(encoder_dict_size,decoder_dict_size, hidden_dim=300,output_size=30)
lstm.train(X_train[:200], y_train[:200],learning_rate=0.001,n_epoch=1000)

print("训练结束,开始预测")
sent="迅速"
sent_word_arr=sent.split(" ")
sent_arr=[]
for i in sent_word_arr:
    sent_arr.append(encoder_dict_words[i])
print (sent_arr)
prd=lstm.predict(sent_arr)
print(prd)

prd_words=[]
for i in prd:
    if i==decoder_dict_words[data_utils.END]:
        break
    prd_words.append(decoder_index_of_words[i])
print ("prd_words:",prd_words)
# sent_str = generate_text(lstm, dict_words, index_of_words)
# print ('Generate sentence:', sent_str)