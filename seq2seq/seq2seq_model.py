# coding=utf-8
import random

import numpy as np
import tensorflow as tf

import data_utils
from jachat.python.ops import rnn_cell, seq2seq


class Seq2SeqModel(object):
    def __init__(self,session,outfile,source_vocab_size,target_vocab_size,buckets,size,num_layers,max_gradient_norm,
                 batch_size,learning_rate,learning_rate_decay_factor,use_lstm=False,num_samples=5,forward_only=False):
        #1.数据定义2+2+8
        self.source_vocab_size=source_vocab_size
        self.target_vocab_size=target_vocab_size

        self.batch_size=batch_size
        self.buckets=buckets

        #size num_layers num_samples
        self.learning_rate=tf.Variable(float(learning_rate),trainable=False)
        self.learning_rate_decay_op=self.learning_rate.assign(self.learning_rate*learning_rate_decay_factor)
        #max_gradient_norm
        #use_lstm forward_only
        self.global_step=tf.Variable(0,trainable=False)

        #2.output_projection和softmax_loss_function定义
        output_projection=None
        softmax_loss_function=None
        if num_samples>0 and num_samples<self.target_vocab_size:
            w=tf.get_variable("proj_w",[size,self.target_vocab_size])
            w_t=tf.transpose(w)
            b=tf.get_variable("proj_b",[self.target_vocab_size])
            output_projection=(w,b)
            def sampled_loss(inputs,labels):
                labels=tf.reshape(labels,[-1,1])
                return tf.nn.sampled_softmax_loss(outfile,w_t,b,inputs,labels,num_samples,self.target_vocab_size)
            softmax_loss_function=sampled_loss

        #3.cell定义
        # single_cell=rnn_cell_xgq.GRUCell(size)
        # if use_lstm:
        cell= rnn_cell.BasicLSTMCell(outfile, size)
        # cell=single_cell
        # if num_layers>1:
        #     cell=rnn_cell_xgq.MultiRNNCell([single_cell]*num_layers)

        #4.seq2seq_f定义
        def seq2seq_f(encoder_inputs,decoder_inputs,do_decode):
            return seq2seq.embedding_attention_seq2seq(session,outfile,
                                                       encoder_inputs, decoder_inputs, cell,
                                                       num_encoder_symbols=source_vocab_size,
                                                       num_decoder_symbols=target_vocab_size,
                                                       embedding_size=size, output_projection=output_projection, feed_previous=do_decode)

        #5.输入变量的定义
        self.encoder_inputs=[]
        self.decoder_inputs=[]
        self.target_weights=[]
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name="encoder{0}".format(i)))
        for i in range(buckets[-1][1]+1):
            self.decoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32,shape=[None],name="weight{0}".format(i)))
        targets=[self.decoder_inputs[i+1] for i in range(len(self.decoder_inputs)-1)]

        #6.输入信息的forward propagation
        if forward_only:
            self.outputs,self.losses= seq2seq.model_with_buckets(outfile, self.encoder_inputs, self.decoder_inputs, targets,
                                                                 self.target_weights, buckets, lambda x,y:seq2seq_f(x,y,True), softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b]=[tf.matmul(output,output_projection[0])+output_projection[1] for output in self.outputs[b]]
        else:
            self.outputs,self.losses= seq2seq.model_with_buckets(outfile, self.encoder_inputs, self.decoder_inputs, targets,
                                                                 self.target_weights, buckets, lambda x,y:seq2seq_f(x,y,False), softmax_loss_function=softmax_loss_function)

        #7.误差信息的backward propagation
        # 返回所有bucket子graph的梯度和SGD更新操作，这些子graph共享输入占位符变量encoder_inputs，区别在于，
       # 对于每一个bucket子图，其输入为该子图对应的长度。
        params=tf.trainable_variables()
        if not forward_only:
            self.gradient_norms=[]
            self.updates=[]
            opt=tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                gradients=tf.gradients(self.losses[b],params)
                clipped_gradients,norm=tf.clip_by_global_norm(gradients,max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients,params),global_step=self.global_step))
        self.saver=tf.train.Saver(tf.all_variables())

    def step(self,outfile, session, encoder_inputs, decoder_inputs, target_weights,bucket_id, forward_only):
        #设置input_feed
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {}
        for l in range(encoder_size):
          input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
          input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
          input_feed[self.target_weights[l].name] = target_weights[l]
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        #设置output_feed
        if not forward_only:
          output_feed = [self.updates[bucket_id],self.gradient_norms[bucket_id],self.losses[bucket_id]]
        else:
          output_feed = [self.losses[bucket_id]]
          for l in range(decoder_size):
            output_feed.append(self.outputs[bucket_id][l])

        #跑程序
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
          return outputs[1], outputs[2], None #GradientNorm,loss,nooutputs
        else:
          return None, outputs[0], outputs[1:] #NoGradientNorm ,loss,outputs


    def get_batch(self, data, bucket_id,current_step):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if current_step==1000:
            print ("get_batch:================================")
            print ("data:",data)
            print ("bucket_id:",bucket_id)
            print ("data[bucket_id]:",data[bucket_id])
            print ("self.buckets[bucket_id]:",self.buckets[bucket_id])
            print ("encoder_size:",encoder_size)
            print ("decoder_size:",decoder_size)
            print ("self.batch_size:",self.batch_size)
        encoder_inputs, decoder_inputs = [], []
        #就算只有2条数据，每个批次多大，就多少次从2条数据中随机取1条生成一个批次
        #decoder_inputs随机的，权值也就随机分布的
        # 哦，第1个位置不算，整体都往后挪动一位
        # encoder_inputs是从前往后补齐0再转置
        # decoder_inputs是先补1再从前往后补齐2

        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +[data_utils.PAD_ID] * decoder_pad_size)
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        for length_idx in range(encoder_size):
          batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]for batch_idx in range(self.batch_size)], dtype=np.int32))
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]for batch_idx in range(self.batch_size)], dtype=np.int32))
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                if length_idx < decoder_size - 1:
                  target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                  batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        if current_step==1000:
            print("batch_encoder_inputs:")
            print(batch_encoder_inputs)
            print("batch_decoder_inputs:")
            print(batch_decoder_inputs)
            print("batch_weights:")
            print(batch_weights)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights



