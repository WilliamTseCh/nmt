from jachat.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables

#被rnn_cell的EmbeddingWrapper(RNNCell)引用到
#被seq2seq的embedding_attention_decoder引用到
#被seq2seq的_extract_argmax_and_embed引用到
def embedding_lookup(outfile,params, ids, partition_strategy="mod", name=None,validate_indices=True, max_norm=None):
  outfile.write("in embedding_ops=============embedding_lookup=================start=====\n")
  print("params:",params)
  if not isinstance(params, list):
    outfile.write("not isinstance(params, list)")
    params = [params]
  with ops.name_scope(name, "embedding_lookup", params + [ids]) as name:
    np = len(params)
    params = ops.convert_n_to_tensor_or_indexed_slices(params, name="params")
    if np == 1:
      with ops.colocate_with(params[0]):
          outfile.write("in embedding_ops=============embedding_lookup======colocate_with=========\n")
          ret = array_ops.gather(outfile,params[0], ids, name=name,validate_indices=validate_indices)
      print("ret:",ret)
      outfile.write("in embedding_ops=============embedding_lookup=================end=====\n")
      return ret

