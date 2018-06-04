# coding=utf-8
from lstmmt import encoder

# 获取数据
encoder_path = r'train.enc'
decoder_path=r'train.dec'
encoder_dict_size = 5
decoder_dict_size = 6
X_train, y_train, encoder_dict_words, encoder_index_of_words,decoder_dict_words, decoder_index_of_words = data_utils.get_vector(encoder_path,decoder_path,encoder_dict_size,decoder_dict_size)
lstm = encoder.myLSTM(encoder_dict_size,decoder_dict_size, hidden_dim=2,output_size=5)
lstm.train(X_train[:200], y_train[:200],learning_rate=0.005,n_epoch=1)

print("训练结束,开始预测")
sent="你"
print ("你")
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