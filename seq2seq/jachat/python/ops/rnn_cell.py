from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from jachat.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from jachat.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.util import nest
#AttributeError: 'Tensor' object has no attribute 'exp'
# def tanh(x):
#     return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))




def _state_size_with_prefix(state_size, prefix=None):
  result_state_size = tensor_shape.as_shape(state_size).as_list()
  if prefix is not None:
    result_state_size = prefix + result_state_size
  return result_state_size


class RNNCell(object):
  #in rnn_xgq rnn use
  def zero_state(self,outfile, batch_size, dtype):
    outfile.write("in rnn_cell_xgq RNNCell zero_state start===================\n")
    state_size = self.state_size
    if nest.is_sequence(state_size):
      print("nest.is_sequence")
      state_size_flat = nest.flatten(state_size)
      zeros_flat = [array_ops.zeros(array_ops.pack(_state_size_with_prefix(s, prefix=[batch_size])),dtype=dtype)for s in state_size_flat]
      for s, z in zip(state_size_flat, zeros_flat):
        z.set_shape(_state_size_with_prefix(s, prefix=[None]))
      zeros = nest.pack_sequence_as(structure=state_size,flat_sequence=zeros_flat)
    print(zeros)
    outfile.write("in rnn_cell_xgq RNNCell zero_state end===================\n")
    return zeros

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
  @property
  def dtype(self):
    (c, h) = self
    return c.dtype


class BasicLSTMCell(RNNCell):
  def __init__(self,outfile, num_units, forget_bias=1.0, input_size=None,state_is_tuple=True, activation=tanh):
    outfile.write("===================== in rnn_cell_xgq   __init__ start==================================\n")
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
  @property
  def state_size(self):
    print("====BasicLSTMCell===state_size====")
    return (LSTMStateTuple(self._num_units, self._num_units)if self._state_is_tuple else 2 * self._num_units)
  @property
  def output_size(self):
    return self._num_units

  def __call__(self,outfile,inputs, state, scope=None):
    outfile.write("===================== in rnn_cell_xgq   BasicLSTMCell __call__ start==================================\n")
    with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      print("==BasicLSTMCell===call==self._state_is_tuple:===========")
      print(self._state_is_tuple)
      if self._state_is_tuple:c, h = state
      concat = _linear(outfile,[inputs, h], 4 * self._num_units, True)
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(1, 4, concat)
      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)
      if self._state_is_tuple:new_state = LSTMStateTuple(new_c, new_h)
      else:new_state = array_ops.concat(1, [new_c, new_h])
      print(new_h)#cant use outfile
      print(new_state)#cant use outfile
      outfile.write("===================== in rnn_cell_xgq   BasicLSTMCell __call__ end==================================\n")
      return new_h, new_state




#seq2seq_xgq  embedding_attention_seq2seq
class EmbeddingWrapper(RNNCell):
  def __init__(self,outfile, cell, embedding_classes, embedding_size, initializer=None):
    outfile.write("===================== in rnn_cell_xgq EmbeddingWrapper  __init__ start==================================\n")
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding_size = embedding_size
    self._initializer = initializer

  @property#cant del
  def state_size(self):
    return self._cell.state_size

  @property#cant del
  def output_size(self):
    return self._cell.output_size

  def __call__(self,outfile, inputs, state, scope=None):
    outfile.write("===================== in rnn_cell_xgq EmbeddingWrapper  __call__ start==================================\n")
    with vs.variable_scope(scope or type(self).__name__):  # "EmbeddingWrapper"
      with ops.device("/cpu:0"):
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)
        data_type = state.dtype
        embedding = vs.get_variable("embedding", [self._embedding_classes, self._embedding_size],initializer=initializer,dtype=data_type)
        embedded = embedding_ops.embedding_lookup(outfile,embedding, array_ops.reshape(inputs, [-1]))
    outfile.write("embedding:")
    print(embedding)#cant use outfile
    outfile.write("embedded:")
    print(embedded)#cant use outfile
    outfile.write("===================== in rnn_cell_xgq EmbeddingWrapper  __call__ end==================================\n")
    return self._cell(outfile,embedded, state)

#BasicLSTMCell use
def _linear(outfile,args, output_size, bias, bias_start=0.0, scope=None):
  outfile.write("===================== in rnn_cell_xgq   _linear start==================================\n")
  if not nest.is_sequence(args):args = [args]
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:total_arg_size += shape[1]
  dtype = [a.dtype for a in args][0]
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable("Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:res = math_ops.matmul(args[0], matrix)
    else:res = math_ops.matmul(array_ops.concat(1, args), matrix)
    bias_term = vs.get_variable("Bias", [output_size],dtype=dtype,initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
  print(bias_term)#cant use outfile
  outfile.write("===================== in rnn_cell_xgq   _linear end==================================\n")
  return res + bias_term


