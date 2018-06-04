# coding=utf-8
import numpy as np
import csv
UNKNOWN = 'UNKNOWN'
START = 'START'
END = 'END'


 # 将文本拆成句子，并加上句子开始和结束标志
def _get_sentences(file_path):
        sents = []
        with open(file_path, 'rt',encoding="utf-8") as f:
            reader = csv.reader(f, skipinitialspace=True)
            sents = [row for row in reader]
            print ('Get {} sentences.'.format(len(sents)))
            return sents

    # 得到每句话的单词，并得到字典及字典中每个词的下标
def _get_dict_wordsIndex(sents):
        sent_words_all=[]
        sent_words=[]
        word_freq={}
        sents_se=[]
        for sent in sents:
            str_sent=sent[0]
            sent_se=[]
            sent_se.append(START)
            sent_word_arr=str_sent.split(" ")
            for w in sent_word_arr:
                sent_se.append(w)
            sent_se.append(END)
            sents_se.append(sent_se)
        for sent in sents_se:
            for w in sent:
                if w!=START and w!=END:
                    sent_words_all.append(w)
        for w in sent_words_all:
            if w in word_freq:
                word_freq[w]+=1
            else:
                word_freq[w]=1
        word_freq[START]=10000
        word_freq[END]=0
        print ('Get {} words.'.format(len(word_freq)))

        common_words = sorted(word_freq, key=word_freq.get, reverse=True)
        common_words.append(UNKNOWN)
        dict_words = dict((word,ix) for ix,word in enumerate(common_words))
        index_of_words=dict((ix,word) for ix,word in enumerate(common_words))
        print ("sents_se:",sents_se)
        print ("dict_words:",dict_words)
        print ("index_of_words:",index_of_words)
        return sents_se, dict_words, index_of_words


    # 得到每句话的单词，并得到字典及字典中每个词的下标
def _get_dict_wordsIndex_encoder(sents):
        sent_words_all=[]
        sent_words=[]
        word_freq={}
        sents_se=[]
        for sent in sents:
            str_sent=sent[0]
            sent_se=[]
            sent_word_arr=str_sent.split(" ")
            sent_se.append(START)
            for w in sent_word_arr:
                sent_se.append(w)
            sent_se.append(END)
            sents_se.append(sent_se)
        for sent in sents_se:
            for w in sent:
                if w!=START and w!=END:
                    sent_words_all.append(w)
        for w in sent_words_all:
            if w in word_freq:
                word_freq[w]+=1
            else:
                word_freq[w]=1
        word_freq[START]=10000
        word_freq[END]=0
        print ('Get {} words.'.format(len(word_freq)))

        common_words = sorted(word_freq, key=word_freq.get, reverse=True)
        common_words.append(UNKNOWN)
        dict_words = dict((word,ix) for ix,word in enumerate(common_words))
        index_of_words=dict((ix,word) for ix,word in enumerate(common_words))
        print ("sents_se:",sents_se)
        print ("dict_words:",dict_words)
        print ("index_of_words:",index_of_words)
        return sents_se, dict_words, index_of_words

    # 得到训练数据
def get_vector(encoder_path,decoder_path):
        encoder_sents = _get_sentences(encoder_path)
        decoder_sents = _get_sentences(decoder_path)
        encoder_sent_words, encoder_dict_words, encoder_index_of_words = _get_dict_wordsIndex_encoder(encoder_sents)
        decoder_sent_words, decoder_dict_words, decoder_index_of_words = _get_dict_wordsIndex(decoder_sents)
        print ("encoder_sent_words:",encoder_sents)
        print ("decoder_sent_words:",decoder_sents)
        # 将每个句子中没包含进词典dict_words中的词替换为unknown_token
        for i, words in enumerate(encoder_sent_words):
            encoder_sent_words[i] = [w if w in encoder_dict_words else UNKNOWN for w in words]
        for i, words in enumerate(decoder_sent_words):
            decoder_sent_words[i] = [w if w in decoder_dict_words else UNKNOWN for w in words]
        print ("将每个句子中没包含进词典encoder_dict_words中的词替换为unknown_token 后encoder_sent_words:",encoder_sent_words)
        print ("将每个句子中没包含进词典decoder_dict_words中的词替换为unknown_token 后decoder_sent_words:",decoder_sent_words)
        X_train = np.array([[encoder_dict_words[w] for w in sent] for sent in encoder_sent_words])
        y_train = np.array([[decoder_dict_words[w] for w in sent] for sent in decoder_sent_words])

        return X_train, y_train, encoder_dict_words, encoder_index_of_words,decoder_dict_words,decoder_index_of_words

def generate_text(rnn, dict_words, index_of_words):
    # dict_words: type list; index_of_words: type dict
    sent = [dict_words[START]]
    # 预测新词，知道句子的结束(END_TOKEN)
    while not sent[-1] == dict_words[END]:
        stats= rnn.forward(sent)
       # pre_y = np.argmax(stats['ys'].reshape(len(x), -1), axis=1)
        sample_word = index_of_words[UNKNOWN]
        # 按预测输出分布进行采样，得到新的词
        #while sample_word == index_of_words[UNKNOWN]:
           #samples = np.random.multinomial(1, next_probs[-1])
           # sample_word = np.argmax(samples)
        # 将新生成的有含义的词(即不为UNKNOWN_TOKEN的词)加入句子
        sent.append(sample_word)
    new_sent = [dict_words[i] for i in sent[1:-1]]
    new_sent_str = ' '.join(new_sent)
    return new_sent_str