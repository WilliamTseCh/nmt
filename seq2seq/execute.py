# coding=utf-8
import math
import os
import sys
import time
import numpy as np
import tensorflow as tf
import data_utils
import seq2seq_model

ResultFile="data/out.txt"
outfile = open(ResultFile, 'w')

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser
gConfig = {}

def get_config(config_file='seq2seq.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    return dict(_conf_ints + _conf_floats + _conf_strings)

_buckets = [(5, 6), (10, 15), (20, 25), (40, 50)]

def read_data(source_path, target_path, max_size=None):
  outfile.write(source_path+"\n")
  outfile.write(target_path+"\n")
  data_set = [[] for _ in _buckets]
  source_file=open(source_path,encoding="utf-8")
  target_file=open(target_path,encoding="utf-8")
  source,target=source_file.readline(),target_file.readline()
  while len(source)>0 and len(target)>0:
        outfile.write (source)
        outfile.write (target)
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            print(data_set)#cant use outfile
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  model = seq2seq_model.Seq2SeqModel(session,outfile,
          gConfig['enc_vocab_size'], gConfig['dec_vocab_size'],
          _buckets, gConfig['layer_size'], gConfig['num_layers'],
          gConfig['max_gradient_norm'], gConfig['batch_size'],
          gConfig['learning_rate'], gConfig['learning_rate_decay_factor'],
          forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(gConfig['working_directory'])
  if ckpt and ckpt.model_checkpoint_path:
    outfile.write("Reading model parameters from %s" % ckpt.model_checkpoint_path+"\n")
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    outfile.write("Created model with fresh parameters.\n")
    session.run(tf.global_variables_initializer())
  return model


def train():
  outfile.write("Preparing data in %s" % gConfig['working_directory']+"\n")
  enc_train, dec_train, enc_dev, dec_dev, _, _ = data_utils.prepare_data(gConfig['working_directory'],
                                                                         gConfig['train_enc'],gConfig['train_dec'],gConfig['test_enc'],gConfig['test_dec'],
                                                                         gConfig['enc_vocab_size'],gConfig['dec_vocab_size'])
  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  with tf.Session(config=config) as sess:
    outfile.write("Creating %d layers of %d units." % (gConfig['num_layers'], gConfig['layer_size']))
    #预处理1
    model = create_model(sess, False)
    outfile.write ("Reading development and training data (limit: %d)."% gConfig['max_train_data_size'])
    #预处理2
    dev_set = read_data(enc_dev, dec_dev)
    train_set = read_data(enc_train, dec_train, gConfig['max_train_data_size'])
    #预处理3
    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]
    # print ("train_bucket_sizes:",train_bucket_sizes)
    # print("train_total_size:",train_total_size)
    # print("train_buckets_scale:",train_buckets_scale)

    #预处理4
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    #while True:
    while current_step<2:
      print ("current_step:",current_step)
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in range(len(train_buckets_scale))if train_buckets_scale[i] > random_number_01])
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id,current_step)
      _, step_loss, _ = model.step(outfile,sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / gConfig['steps_per_checkpoint']
      loss += step_loss / gConfig['steps_per_checkpoint']
      current_step += 1
      if current_step % gConfig['steps_per_checkpoint'] == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        checkpoint_path = os.path.join(gConfig['working_directory'], "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        for bucket_id in range(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            outfile.write("  eval: empty bucket %d" % (bucket_id))
            outfile.write("\n")
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id,current_step)
          _, eval_loss, output_logits = model.step(outfile,sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
          outputs=[]
          for logit in output_logits:
              print("logit:")
              print(logit)
              _mx=np.argmax(logit, axis=1)
              print("_mx:")
              print(_mx)
              outputs.append(_mx)
          print(outputs)#cant use outfile
          batch_size=len(outputs[0])
          decoder_size=len(decoder_inputs)
          decoder_inputs_t=[]
          preds=[]
          print("decoder_inputs:")
          print(decoder_inputs)
          for batch_idx in range(batch_size):
            decoder_inputs_t.append(np.array([decoder_inputs[length_idx][batch_idx]for length_idx in range(decoder_size)], dtype=np.int32))
          print("decoder_inputs_t:")
          print(decoder_inputs_t)
          for batch_idx in range(batch_size):
            preds.append(np.array([outputs[length_idx][batch_idx]for length_idx in range(decoder_size)], dtype=np.int32))
          print("preds:")
          print(preds)
          sum_cnt=batch_size
          print("sum_cnt:")
          print(sum_cnt)
          correct_pred=0
          for batch_idx in range(batch_size):
              flag=True
              for length_idx in range(decoder_size):
                  if(decoder_inputs_t[batch_idx][length_idx]!=preds[batch_idx][length_idx]):
                      flag=False
              if flag:
                  correct_pred+=1
          print("correct_pred:")
          print(correct_pred)#cant use outfile
          precise=correct_pred/sum_cnt
          print("precise:")
          print(precise)#cant use outfile
          print("output_logits:")
          print(output_logits)#cant use outfile
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          outfile.write("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()

def decode():
  with tf.Session() as sess:
    model = create_model(sess, True)
    model.batch_size = 1
    enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.dec" % gConfig['dec_vocab_size'])
    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    print ("enc_vocab:",enc_vocab,"\n")
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)
    print("rev_dec_vocab:",rev_dec_vocab,"\n")
    sys.stdout.write(">")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      print ("sentence:",sentence)
      token_ids = data_utils.sentence_to_token_ids(sentence, enc_vocab)
      print ("token_ids:",token_ids)
      bucket_id = min([b for b in range(len(_buckets)) if _buckets[b][0] > len(token_ids)])
      print ("bucket_id:",bucket_id)
      encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
      _, _, output_logits = model.step(outfile, sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
      print ("output_logits:",output_logits)
      outputs=[]
      for logit in output_logits:
          print ("logit:",logit)
          _mx=np.argmax(logit, axis=1)
          print ("_mx:",_mx)
          print ("int(_mx):",int(_mx))
          outputs.append(int(_mx))
      print ("outputs:",outputs)
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      res_str=""
      for output in outputs:
          res_str+=rev_dec_vocab[output][0]+" "
      print (res_str)
      print(">", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

if __name__ == '__main__':
    gConfig = get_config()
    print('\n>> Mode : %s\n' %(gConfig['mode']))
    if gConfig['mode'] == 'train':
        train()
    elif gConfig['mode'] == 'test':
        decode()
