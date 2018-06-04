"""
"""
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import sparse_ops
from jachat.python.ops import embedding_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.nn_ops import *
from tensorflow.python.ops.candidate_sampling_ops import *
from jachat.python.ops.embedding_ops import *
from jachat.python.ops.rnn import *

def _sum_rows(x):
  cols = array_ops.shape(x)[1]
  ones_shape = array_ops.pack([cols, 1])
  ones = array_ops.ones(ones_shape, x.dtype)
  return array_ops.reshape(math_ops.matmul(x, ones), [-1])


def _compute_sampled_logits(outfile,weights,biases,inputs,labels,num_sampled,num_classes,
                            num_true=1,sampled_values=None,subtract_log_q=True,remove_accidental_hits=False,partition_strategy="mod",name=None):
  if not isinstance(weights, list):
    weights = [weights]
  with ops.name_scope(name, "compute_sampled_logits",weights + [biases, inputs, labels]):
    if labels.dtype != dtypes.int64:
      labels = math_ops.cast(labels, dtypes.int64)
    labels_flat = array_ops.reshape(labels, [-1])
    if sampled_values is None:
      sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(true_classes=labels,num_true=num_true,num_sampled=num_sampled,unique=True,range_max=num_classes)
    sampled, true_expected_count, sampled_expected_count = sampled_values
    all_ids = array_ops.concat(0, [labels_flat, sampled])
    all_w = embedding_ops.embedding_lookup(outfile,weights, all_ids, partition_strategy=partition_strategy)
    all_b = embedding_ops.embedding_lookup(outfile,biases, all_ids)
    true_w = array_ops.slice(all_w, [0, 0], array_ops.pack([array_ops.shape(labels_flat)[0], -1]))
    true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat(0, [[-1, num_true], dim])
    row_wise_dots = math_ops.mul(array_ops.expand_dims(inputs, 1),array_ops.reshape(true_w, new_true_w_shape))
    dots_as_matrix = array_ops.reshape(row_wise_dots,array_ops.concat(0, [[-1], dim]))
    true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
    true_b = array_ops.reshape(true_b, [-1, num_true])
    true_logits += true_b
    sampled_w = array_ops.slice(all_w, array_ops.pack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
    sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])
    sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True) + sampled_b
    if remove_accidental_hits:
      acc_hits = candidate_sampling_ops.compute_accidental_hits(labels, sampled, num_true=num_true)
      acc_indices, acc_ids, acc_weights = acc_hits
      acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
      acc_ids_2d_int32 = array_ops.reshape(math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
      sparse_indices = array_ops.concat(1, [acc_indices_2d, acc_ids_2d_int32],"sparse_indices")
      sampled_logits_shape = array_ops.concat(0,[array_ops.shape(labels)[:1], array_ops.expand_dims(num_sampled, 0)])
      if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
      sampled_logits += sparse_ops.sparse_to_dense(sparse_indices,sampled_logits_shape,acc_weights,default_value=0.0,validate_indices=False)
    if subtract_log_q:
      true_logits -= math_ops.log(true_expected_count)
      sampled_logits -= math_ops.log(sampled_expected_count)
    out_logits = array_ops.concat(1, [true_logits, sampled_logits])
    out_labels = array_ops.concat(1,[array_ops.ones_like(true_logits) / num_true,array_ops.zeros_like(sampled_logits)])
  return out_logits, out_labels


def sampled_softmax_loss(outfile,weights,biases,inputs,labels,num_sampled,num_classes,num_true=1,sampled_values=None,remove_accidental_hits=True,partition_strategy="mod",name="sampled_softmax_loss"):
  print("sampled_softmax_loss==================")
  logits, labels = _compute_sampled_logits(outfile,weights,biases,inputs,labels,num_sampled,
      num_classes,num_true=num_true,sampled_values=sampled_values,subtract_log_q=True,
      remove_accidental_hits=remove_accidental_hits,partition_strategy=partition_strategy,name=name)
  sampled_losses = nn_ops.softmax_cross_entropy_with_logits(logits, labels)
  return sampled_losses

