from jachat.python.ops import rnn_cell
from tensorflow.python import shape
from jachat.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from jachat.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

from jachat.python.ops import rnn

linear = rnn_cell._linear

#embedding_attention_seq2seq
#attention_decoder
#embedding_attention_decoder
#_extract_argmax_and_embed
#sequence_loss
#sequence_loss_by_example
#model_with_buckets

#in Seq2SeqModel __init__ use
def embedding_attention_seq2seq(session,outfile,encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,
                                embedding_size,num_heads=1,
                                output_projection=None,feed_previous=False,
                                dtype=None,scope=None,initial_state_attention=False):
  outfile.write("in seq2seq===embedding_attention_seq2seq start====================\n")
  with variable_scope.variable_scope(scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    encoder_cell = rnn_cell.EmbeddingWrapper(outfile, cell, embedding_classes=num_encoder_symbols, embedding_size=embedding_size)
    print("embedding_attention_seq2seq encoder_inputs:")
    # print(session.run(encoder_inputs))
    encoder_outputs, encoder_state = rnn.rnn(outfile, encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)

    # Decoder.
    output_size = None
    if isinstance(feed_previous, bool):
      return embedding_attention_decoder(outfile,
          decoder_inputs,encoder_state,attention_states,
          cell,num_decoder_symbols,embedding_size,
          num_heads=num_heads,output_size=output_size,
          output_projection=output_projection,feed_previous=feed_previous,
          initial_state_attention=initial_state_attention)




#above embedding_attention_seq2seq use
def embedding_attention_decoder(outfile,decoder_inputs,initial_state,attention_states,cell,
                                num_symbols,embedding_size,num_heads=1,
                                output_size=None,output_projection=None,
                                feed_previous=False,update_embedding_for_previous=True,
                                dtype=None,scope=None,initial_state_attention=False):
  outfile.write("in seq2seq_xgq===========embedding_attention_decoder start================\n")
  with variable_scope.variable_scope(scope or "embedding_attention_decoder", dtype=dtype) as scope:
    embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(outfile,embedding, output_projection,update_embedding_for_previous) if feed_previous else None

    emb_inp = [embedding_ops.embedding_lookup(outfile,embedding, i) for i in decoder_inputs]
    return attention_decoder(outfile,emb_inp,initial_state,attention_states,
        cell,output_size=cell.output_size,num_heads=num_heads,
        loop_function=loop_function,initial_state_attention=initial_state_attention)

#above embedding_attention_decoder use
def attention_decoder(outfile,decoder_inputs,initial_state,attention_states,cell,
                      output_size=None,num_heads=1,loop_function=None,
                      dtype=None,scope=None,initial_state_attention=False):
  outfile.write("in seq2seq_xgq===========attention_decoder start================\n")
  with variable_scope.variable_scope(scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype
    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value
    outfile.write("To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.\n")
    hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    outfile.write("Size of query vectors for attention.\n")
    attention_vec_size = attn_size
    for a in range(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,[1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))
    state = initial_state
    def attention(query):
      outfile.write("Put attention masks on hidden using hidden_features and query.\n")
      outfile.write("ds:Results of attention reads will be stored here.\n")
      ds = []
      query_list = nest.flatten(query)
      query = array_ops.concat(1, query_list)
      for a in range(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          y = linear(outfile,query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,[1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds
    outputs = []
    prev = None
    batch_attn_size = array_ops.pack([batch_size, attn_size])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype)for _ in range(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns = attention(initial_state)
    for i, inp in enumerate(decoder_inputs):
      if i > 0:variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      x = linear(outfile,[inp] + attns, input_size, True)
      # Run the RNN.
      cell_output, state = cell(outfile,x, state)
      # Run the attention mechanism.
      attns = attention(state)
      print("in attention(query) attns:",attns)
      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear(outfile,[cell_output] + attns, cell.output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)
  return outputs, state


# embedding_attention_decoder use
def _extract_argmax_and_embed(outfile,embedding, output_projection=None,update_embedding=True):
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    emb_prev = embedding_ops.embedding_lookup(outfile,embedding, prev_symbol)
    return emb_prev
  return loop_function

#in Seq2SeqModel __init__ use
def model_with_buckets(outfile,encoder_inputs, decoder_inputs, targets, weights,buckets, seq2seq, softmax_loss_function=None,per_example_loss=False, name=None):
  outfile.write("in seq2seq_xgq===========model_with_buckets start================\n")
  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        losses.append(sequence_loss(outfile,outputs[-1], targets[:bucket[1]], weights[:bucket[1]],softmax_loss_function=softmax_loss_function))
  outfile.write("================in seq2seq_xgq===========model_with_buckets end\n")
  return outputs, losses

#model_with_buckets use
def sequence_loss(outfile,logits, targets, weights,softmax_loss_function=None, name=None):
  outfile.write("in seq2seq_xgq===========sequence_loss start================\n")
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(sequence_loss_by_example(outfile,logits, targets, weights,softmax_loss_function=softmax_loss_function))
    batch_size = array_ops.shape(targets[0])[0]
    return cost / math_ops.cast(batch_size, cost.dtype)

#model_with_buckets use
def sequence_loss_by_example(outfile,logits, targets, weights,softmax_loss_function=None, name=None):
  outfile.write("in seq2seq_xgq===========sequence_loss_by_example start================\n")
  with ops.name_scope(name, "sequence_loss_by_example",logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    total_size = math_ops.add_n(weights)
    total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
    log_perps /= total_size
  return log_perps
























