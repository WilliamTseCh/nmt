# coding=utf-8

import os

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def tokenizer(sentence):
  words = sentence.replace("\n","").split(" ")
  return words


def create_vocabulary(data_path,vocabulary_path,max_vocabulary_size):
    print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
    vocab = {}
    if not os.path.exists(vocabulary_path):
        infile=open(data_path,encoding="utf-8")
        line=infile.readline().replace("\n","")
        while len(line)>0:
            words =tokenizer(line)
            for word in words:
              if word in vocab:
                vocab[word] += 1
              else:
                vocab[word] = 1
            line=infile.readline().replace("\n","")
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print('>> Full Vocabulary Size :',len(vocab_list))
        print("vocab_list:",vocab_list)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        vocab_file = open(vocabulary_path, 'w',encoding="utf-8")
        for w in vocab_list:
            vocab_file.write(str(w) +"\n")

def initialize_vocabulary(vocabulary_path):
    rev_vocab = {}
    infile=open(vocabulary_path,encoding="utf-8")
    line=infile.readline().replace("\n","")
    count=0
    while len(line)>0:
        rev_vocab[count]=line
        line=infile.readline().replace("\n","")
        count+=1
    vocab={}
    for x in rev_vocab.keys():
        vocab[rev_vocab[x]]=x
    return vocab, rev_vocab


def sentence_to_token_ids(sentence, vocabulary):
    words = tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]

def data_to_token_ids(data_path, token_ids_path, vocabulary_path):
  if not os.path.exists(token_ids_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    data_file=open(data_path,encoding="utf-8")
    token_ids_file=open(token_ids_path,"w",encoding="utf-8")
    line=data_file.readline().replace("\n","")
    while len(line)>0:
          token_ids = sentence_to_token_ids(line, vocab)
          token_ids_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
          print (" ".join([str(tok) for tok in token_ids]) + "\n")
          line=data_file.readline().replace("\n","")

def prepare_data(working_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size):
    print ("================prepare_custom_data start:==============================================")
    # Create vocabularies of the appropriate sizes.
    enc_vocab_path = os.path.join(working_directory, "vocab%d.enc" % enc_vocabulary_size)
    dec_vocab_path = os.path.join(working_directory, "vocab%d.dec" % dec_vocabulary_size)
    create_vocabulary(train_enc,enc_vocab_path, enc_vocabulary_size)
    create_vocabulary(train_dec,dec_vocab_path,  dec_vocabulary_size)

    # Create token ids for the training data.
    enc_train_ids_path = train_enc + (".ids%d" % enc_vocabulary_size)
    dec_train_ids_path = train_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(train_enc, enc_train_ids_path, enc_vocab_path)
    data_to_token_ids(train_dec, dec_train_ids_path, dec_vocab_path)

    # Create token ids for the development data.
    enc_test_ids_path = test_enc + (".ids%d" % enc_vocabulary_size)
    dec_test_ids_path = test_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(test_enc, enc_test_ids_path, enc_vocab_path)
    data_to_token_ids(test_dec, dec_test_ids_path, dec_vocab_path)

    print ("================prepare_custom_data end:==============================================")
    return (enc_train_ids_path, dec_train_ids_path, enc_test_ids_path, dec_test_ids_path, enc_vocab_path, dec_vocab_path)
