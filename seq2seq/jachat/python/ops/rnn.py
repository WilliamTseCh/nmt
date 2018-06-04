from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

#in seq2seq_xgq embedding_attention_seq2seq use
def rnn(outfile,cell, inputs, initial_state=None, dtype=None,sequence_length=None, scope=None):
  outfile.write("=====================  rnn_xgq   rnn start==================================\n")
  outputs = []
  with vs.variable_scope(scope or "RNN") as varscope:
    first_input = inputs
    while nest.is_sequence(first_input):
      first_input = first_input[0]
    batch_size = array_ops.shape(first_input)[0]
    state = cell.zero_state(outfile,batch_size, dtype)
    for time, input_ in enumerate(inputs):
      if time > 0: varscope.reuse_variables()
      call_cell = lambda: cell(outfile,input_, state)
      (output, state) = call_cell()
      outputs.append(output)
    print(outputs)#cant use outfile
    outfile.write("=====================  rnn_xgq   rnn end==================================\n")
    return (outputs, state)