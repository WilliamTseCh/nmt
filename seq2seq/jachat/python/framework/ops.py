from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import copy
import linecache
import re
import sys
import threading

import six
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import versions
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils

ResultFile="data/out_ops.txt"
outfile = open(ResultFile, 'w')

def _override_helper(clazz_object, operator, func):
  setattr(clazz_object, operator, func)

def _convert_stack(stack):
  ret = []
  for filename, lineno, name, frame_globals in stack:
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, frame_globals)
    if line:
      line = line.strip()
    else:
      line = None
    ret.append((filename, lineno, name, line))
  return ret

def _extract_stack():
  try:
    raise ZeroDivisionError
  except ZeroDivisionError:
    f = sys.exc_info()[2].tb_frame.f_back
  ret = []
  while f is not None:
    lineno = f.f_lineno
    co = f.f_code
    filename = co.co_filename
    name = co.co_name
    frame_globals = f.f_globals
    ret.append((filename, lineno, name, frame_globals))
    f = f.f_back
  ret.reverse()
  return ret


def _as_graph_element(obj):
  conv_fn = getattr(obj, "_as_graph_element", None)
  if conv_fn and callable(conv_fn):
    return conv_fn()
  return None

_TENSOR_LIKE_TYPES = tuple()
def is_dense_tensor_like(t):
  return isinstance(t, _TENSOR_LIKE_TYPES)

def register_dense_tensor_like_type(tensor_type):
  global _TENSOR_LIKE_TYPES
  _TENSOR_LIKE_TYPES = tuple(list(_TENSOR_LIKE_TYPES) + [tensor_type])

class _TensorLike(object):pass
class Tensor(_TensorLike):
  OVERLOADABLE_OPERATORS = {
      # Binary.
      "__add__",
      "__radd__",
      "__sub__",
      "__rsub__",
      "__mul__",
      "__rmul__",
      "__div__",
      "__rdiv__",
      "__truediv__",
      "__rtruediv__",
      "__floordiv__",
      "__rfloordiv__",
      "__mod__",
      "__rmod__",
      "__lt__",
      "__le__",
      "__gt__",
      "__ge__",
      "__and__",
      "__rand__",
      "__or__",
      "__ror__",
      "__xor__",
      "__rxor__",
      "__getitem__",
      "__pow__",
      "__rpow__",
      # Unary.
      "__invert__",
      "__neg__",
      "__abs__"
  }

  def __init__(self, op, value_index, dtype):
    self._op = op
    self._value_index = value_index
    self._dtype = dtypes.as_dtype(dtype)
    self._shape = tensor_shape.unknown_shape()
    self._consumers = []
    self._handle_shape = tensor_shape_pb2.TensorShapeProto()
    self._handle_dtype = types_pb2.DT_INVALID
  @property
  def op(self):
    return self._op
  @property
  def dtype(self):
    return self._dtype
  @property
  def graph(self):
    return self._op.graph
  @property
  def name(self):
    return "%s:%d" % (self._op.name, self._value_index)
  @property
  def device(self):
    return self._op.device
  def _shape_as_list(self):
    if self._shape.ndims is not None:
      return [dim.value for dim in self._shape.dims]
    else:
      return None

  def get_shape(self):
    return self._shape
  def set_shape(self, shape):
    self._shape = self._shape.merge_with(shape)
  @property
  def value_index(self):
    return self._value_index
  def consumers(self):
    return self._consumers
  def _add_consumer(self, consumer):
    self._consumers.append(consumer)

  def _as_node_def_input(self):
    if self._value_index == 0:
      return self._op.name
    else:
      return "%s:%d" % (self._op.name, self._value_index)

  def __str__(self):
    return "Tensor(\"%s\"%s%s%s)" % (
        self.name,
        (", shape=%s" % self.get_shape())
        if self.get_shape().ndims is not None else "",
        (", dtype=%s" % self._dtype.name) if self._dtype else "",
        (", device=%s" % self.device) if self.device else "")

  def __repr__(self):
    return "<tf.Tensor '%s' shape=%s dtype=%s>" % (
        self.name, self.get_shape(), self._dtype.name)

  def __hash__(self):
    return id(self)

  def __eq__(self, other):
    return id(self) == id(other)
  __array_priority__ = 100

  @staticmethod
  def _override_operator(operator, func):
    _override_helper(Tensor, operator, func)

  def __iter__(self):
    raise TypeError("'Tensor' object is not iterable.")

  def __bool__(self):
    raise TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed. "
                    "Use `if t is not None:` instead of `if t:` to test if a "
                    "tensor is defined, and use TensorFlow ops such as "
                    "tf.cond to execute subgraphs conditioned on the value of "
                    "a tensor.")

  def __nonzero__(self):
    raise TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed. "
                    "Use `if t is not None:` instead of `if t:` to test if a "
                    "tensor is defined, and use TensorFlow ops such as "
                    "tf.cond to execute subgraphs conditioned on the value of "
                    "a tensor.")

  def eval(self, feed_dict=None, session=None):
    outfile.write("in python/framework/ops========eval==========================\n")
    return _eval_using_default_session(self, feed_dict, self.graph, session)

def _TensorTensorConversionFunction(t, dtype=None, name=None, as_ref=False):
  _ = name, as_ref
  if dtype and not dtype.is_compatible_with(t.dtype):
    raise ValueError(
        "Tensor conversion requested dtype %s for Tensor with dtype %s: %r"
        % (dtype.name, t.dtype.name, str(t)))
  return t

_tensor_conversion_func_registry = {0: [(Tensor, _TensorTensorConversionFunction)]}
register_dense_tensor_like_type(Tensor)

#被convert_to_tensor_or_indexed_slices引用
#被convert_n_to_tensor引用
def convert_to_tensor(value,dtype=None,name=None,as_ref=False,preferred_dtype=None):
  outfile.write("in python/framework/ops======convert_to_tensor=========\n")
  error_prefix = "" if name is None else "%s: " % name
  if dtype is not None:
    dtype = dtypes.as_dtype(dtype)
  for _, funcs_at_priority in sorted(_tensor_conversion_func_registry.items()):
    for base_type, conversion_func in funcs_at_priority:
      if isinstance(value, base_type):
        ret = None
        if dtype is None and preferred_dtype is not None:
          try:
            ret = conversion_func(value, dtype=preferred_dtype, name=name, as_ref=as_ref)
          except (TypeError, ValueError):
            ret = None
        if ret is None:
          ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
        if ret is NotImplemented:
          continue
        return ret

#被op_def_lib引用
def convert_n_to_tensor(values,dtype=None,name=None,as_ref=False,preferred_dtype=None):
  outfile.write("in python/framework/ops======convert_n_to_tensor=========\n")
  ret = []
  for i, value in enumerate(values):
    n = None if name is None else "%s_%d" % (name, i)
    ret.append(convert_to_tensor(value,dtype=dtype,name=n,as_ref=as_ref,preferred_dtype=preferred_dtype))
  return ret

#被convert_n_to_tensor_or_indexed_slices引用
#被colocate_with引用
def convert_to_tensor_or_indexed_slices(value, dtype=None, name=None,as_ref=False):
  outfile.write("in python/framework/ops======convert_to_tensor_or_indexed_slices=========\n")
  if isinstance(value, _TensorLike):
    return value
  else:
    return convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)

#被embedding_lookup引用
def convert_n_to_tensor_or_indexed_slices(values, dtype=None, name=None,as_ref=False):
  outfile.write("in python/framework/ops======convert_n_to_tensor_or_indexed_slices=========\n")
  ret = []
  for i, value in enumerate(values):
    if value is None:
      ret.append(value)
    else:
      n = None if name is None else "%s_%d" % (name, i)
      ret.append(convert_to_tensor_or_indexed_slices(value, dtype=dtype, name=n,as_ref=as_ref))
  return ret


def register_tensor_conversion_function(base_type, conversion_func,priority=100):
  outfile.write("in python/framework/ops======register_tensor_conversion_function=========\n")
  #这个try except不能删除
  try:
    funcs_at_priority = _tensor_conversion_func_registry[priority]
  except KeyError:
    funcs_at_priority = []
    _tensor_conversion_func_registry[priority] = funcs_at_priority
  funcs_at_priority.append((base_type, conversion_func))

#被session的_get_feeds_for_indexed_slices引用
class IndexedSlices(_TensorLike):
  def __init__(self, values, indices, dense_shape=None):
    outfile.write("in python/framework/ops======IndexedSlices=========\n")
    _get_graph_from_inputs([values, indices, dense_shape])
    self._values = values
    self._indices = indices
    self._dense_shape = dense_shape
  @property
  def values(self):
    return self._values
  @property
  def indices(self):
    return self._indices
  @property
  def dense_shape(self):
    return self._dense_shape
  @property
  def name(self):
    return self.values.name
  @property
  def device(self):
    return self.values.device
  @property
  def op(self):
    return self.values.op
  @property
  def dtype(self):
    return self.values.dtype
  @property
  def graph(self):
    return self._values.graph
  def __str__(self):
    return "IndexedSlices(indices=%s, values=%s%s)" % (self._indices, self._values,(", dense_shape=%s" % self._dense_shape)if self._dense_shape is not None else "")
  def __neg__(self):
    return IndexedSlices(-self.values, self.indices, self.dense_shape)

IndexedSlicesValue = collections.namedtuple("IndexedSlicesValue", ["values", "indices", "dense_shape"])

def _device_string(dev_spec):
  outfile.write("in python/framework/ops======_device_string=====被Operation引用====\n")
  if isinstance(dev_spec, pydev.DeviceSpec):
    return dev_spec.to_string()
  else:
    return dev_spec

def _NodeDef(op_type, name, device=None, attrs=None):
  outfile.write("in python/framework/ops======_NodeDef=====被create_op引用====\n")
  node_def = node_def_pb2.NodeDef()
  node_def.op = compat.as_bytes(op_type)
  node_def.name = compat.as_bytes(name)
  if attrs is not None:
    for k, v in six.iteritems(attrs):
      node_def.attr[k].CopyFrom(v)
  if device is not None:
    if callable(device):
      node_def.device = device(node_def)
    else:
      node_def.device = _device_string(device)
  return node_def

_VALID_OP_NAME_REGEX = re.compile("^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$")
_VALID_SCOPE_NAME_REGEX = re.compile("^[A-Za-z0-9_.\\-/]*$")

class Operation(object):
  def __init__(self, node_def, g, inputs=None, output_types=None,control_inputs=None, input_types=None, original_op=None,op_def=None):
    outfile.write("in python/framework/ops===Operation===__init__=========\n")
    self._node_def = copy.deepcopy(node_def)
    self._graph = g
    if inputs is None:
      inputs = []
    self._inputs = list(inputs)
    for a in self._inputs:
      a._add_consumer(self)
    if output_types is None:
      output_types = []
    self._output_types = output_types
    self._outputs = [Tensor(self, i, output_type)for i, output_type in enumerate(output_types)]
    if input_types is None:
      input_types = [i.dtype.base_dtype for i in self._inputs]
    self._input_types = input_types
    self._control_inputs = []
    if control_inputs:
      for c in control_inputs:
        c_op = None
        if isinstance(c, Operation):
          c_op = c
        elif isinstance(c, (Tensor, IndexedSlices)):
          c_op = c.op
        self._control_inputs.append(c_op)
    self._original_op = original_op
    self._op_def = op_def
    self._traceback = _extract_stack()
    self._control_flow_context = g._get_control_flow_context()
    if self._control_flow_context is not None:
      self._control_flow_context.AddOp(self)
    self._id_value = self._graph._next_id()
    self._recompute_node_def()

  def colocation_groups(self):
    default_colocation_group = [compat.as_bytes("loc:@%s" % self._node_def.name)]
    if "_class" not in self._node_def.attr:
      return default_colocation_group
    attr_groups = [class_name for class_name in self.get_attr("_class") if class_name.startswith(b"loc:@")]
    return attr_groups if attr_groups else default_colocation_group

  def values(self):
    return tuple(self.outputs)
  def _get_control_flow_context(self):
    return self._control_flow_context
  def _set_control_flow_context(self, context):
    self._control_flow_context = context
  @property
  def name(self):
    return self._node_def.name
  @property
  def _id(self):
    return self._id_value
  @property
  def device(self):
    return self._node_def.device
  def _set_device(self, device):
    self._node_def.device = _device_string(device)
  def _add_input(self, tensor, dtype=None):
    if dtype is None:
      dtype = tensor.dtype
    else:
      dtype = dtypes.as_dtype(dtype)
    self._inputs.append(tensor)
    self._input_types.append(dtype)
    tensor._add_consumer(self)
    self._recompute_node_def()

  def _update_input(self, index, tensor, dtype=None):
    if dtype is None:
      dtype = tensor.dtype
    else:
      dtype = dtypes.as_dtype(dtype)
    self._inputs[index].consumers().remove(self)
    self._inputs[index] = tensor
    self._input_types[index] = dtype
    tensor._add_consumer(self)
    self._recompute_node_def()

  def _add_control_inputs(self, ops):
    if ops:
      for op in ops:
        self._control_inputs.append(op)
      self._recompute_node_def()

  def _add_control_input(self, op):
    self._add_control_inputs([op])
  def _recompute_node_def(self):
    del self._node_def.input[:]
    self._node_def.input.extend([t._as_node_def_input() for t in self._inputs])
    if self._control_inputs:
      self._node_def.input.extend(["^%s" % op.name for op in self._control_inputs])

  def __str__(self):
    return str(self._node_def)
  @property
  def outputs(self):
    return self._outputs

  class _InputList(object):
    def __init__(self, op):
      self._op = op
    def __iter__(self):
      return iter(self._op._inputs)
    def __len__(self):
      return len(self._op._inputs)
    def __bool__(self):
      return bool(self._op._inputs)
    __nonzero__ = __bool__
    def __getitem__(self, i):
      return self._op._inputs[i]
  @property
  def inputs(self):
    return Operation._InputList(self)
  @property
  def _input_dtypes(self):
    return self._input_types
  @property
  def control_inputs(self):
    return self._control_inputs
  @property
  def type(self):
    return self._node_def.op
  @property
  def graph(self):
    return self._graph
  @property
  def node_def(self):
    return self._node_def

  @property
  def op_def(self):
    return self._op_def

  @property
  def traceback(self):
    return _convert_stack(self._traceback)

  def get_attr(self, name):
    fields = ["s", "i", "f", "b", "type", "shape", "tensor"]
    if name not in self._node_def.attr:
      raise ValueError("No attr named '" + name + "' in " +
                       str(self._node_def))
    x = self._node_def.attr[name]
    # Treat an empty oneof value as an empty list.
    if not x.WhichOneof("value"):
      return []
    if x.HasField("list"):
      for f in fields:
        if getattr(x.list, f):
          return list(getattr(x.list, f))
      return []
    else:
      for f in fields:
        if x.HasField(f):
          return getattr(x, f)
      assert False, "Unsupported field type in " + str(x)

_gradient_registry = registry.Registry("gradient")

class RegisterGradient(object):
  def __init__(self, op_type):
    outfile.write("in python/framework/ops===RegisterGradient===__init__=======\n")
    self._op_type = op_type
  def __call__(self, f):
    _gradient_registry.register(f, self._op_type)
    return f

#被array_grad引用
def NotDifferentiable(op_type):
  outfile.write("in python/framework/ops===NotDifferentiable=========\n")
  _gradient_registry.register(None, op_type)
NoGradient = NotDifferentiable

def get_gradient_function(op):
  """Returns the function that computes gradients for "op"."""
  if not op.inputs: return None
  try:
    op_type = op.get_attr("_gradient_op_type")
  except ValueError:
    op_type = op.type
  return _gradient_registry.lookup(op_type)

_shape_registry = registry.Registry("shape functions")
_default_shape_function_registry = registry.Registry("default shape functions")

_call_cpp_shape_fn = None
_call_cpp_shape_fn_and_require_op = None

def _set_call_cpp_shape_fn(call_cpp_shape_fn):
  global _call_cpp_shape_fn, _call_cpp_shape_fn_and_require_op
  if _call_cpp_shape_fn:
    return
  def call_without_requiring(op):
    return call_cpp_shape_fn(op, require_shape_fn=False)
  _call_cpp_shape_fn = call_without_requiring
  def call_with_requiring(op):
    return call_cpp_shape_fn(op, require_shape_fn=True)
  _call_cpp_shape_fn_and_require_op = call_with_requiring


def set_shapes_for_outputs(op):
  outfile.write("in python/framework/ops=====set_shapes_for_outputs===被create_op引用====\n")
  try:
    shape_func = _shape_registry.lookup(op.type)
  except LookupError:
    try:
      shape_func = _default_shape_function_registry.lookup(op.type)
    except LookupError:
      shape_func = _call_cpp_shape_fn_and_require_op
  shapes = shape_func(op)
  if isinstance(shapes, dict):
    shapes_dict = shapes
    shapes = shapes_dict["shapes"]
    handle_shapes = shapes_dict["handle_shapes"]
    handle_dtypes = shapes_dict["handle_dtypes"]
    for output, handle_shape, handle_dtype in zip(op.outputs, handle_shapes, handle_dtypes):
      output._handle_shape = handle_shape
      output._handle_dtype = handle_dtype
  for output, s in zip(op.outputs, shapes):
    output.set_shape(s)
_stats_registry = registry.Registry("statistical functions")

class RegisterStatistics(object):
  def __init__(self, op_type, statistic_type):
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string.")
    if "," in op_type:
      raise TypeError("op_type must not contain a comma.")
    self._op_type = op_type
    if not isinstance(statistic_type, six.string_types):
      raise TypeError("statistic_type must be a string.")
    if "," in statistic_type:
      raise TypeError("statistic_type must not contain a comma.")
    self._statistic_type = statistic_type
  def __call__(self, f):
    _stats_registry.register(f, self._op_type + "," + self._statistic_type)
    return f

class Graph(object):
  def __init__(self):
    outfile.write("in python/framework/ops Graph=============__init__=====start=====\n")
    self._lock = threading.Lock()
    self._nodes_by_id = dict()  # GUARDED_BY(self._lock)
    self._next_id_counter = 0  # GUARDED_BY(self._lock)
    self._nodes_by_name = dict()  # GUARDED_BY(self._lock)
    self._version = 0  # GUARDED_BY(self._lock)
    self._name_stack = ""
    self._names_in_use = {}
    self._device_function_stack = []
    self._default_original_op = None
    self._control_flow_context = None
    self._control_dependencies_stack = []
    self._collections = {}
    self._seed = None
    self._attr_scope_map = {}
    self._op_to_kernel_label_map = {}
    self._gradient_override_map = {}
    self._finalized = False
    self._functions = collections.OrderedDict()
    self._graph_def_versions = versions_pb2.VersionDef(producer=versions.GRAPH_DEF_VERSION,min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER)
    self._building_function = False
    self._colocation_stack = []
    self._unfeedable_tensors = set()
    self._unfetchable_ops = set()
    self._handle_feeders = {}
    self._handle_readers = {}
    self._handle_movers = {}
    self._handle_deleters = {}
    self._container = ""
    self._registered_ops = op_def_registry.get_registered_ops()
    outfile.write("in python/framework/ops Graph=============__init__====end======\n")

  def _check_not_finalized(self):
    if self._finalized:
      raise RuntimeError("Graph is finalized and cannot be modified.")

  def _add_op(self, op):
    outfile.write("in python/framework/ops Graph====_add_op======\n")
    self._check_not_finalized()
    with self._lock:
      self._nodes_by_id[op._id] = op
      self._nodes_by_name[op.name] = op
      self._version = max(self._version, op._id)


  @property
  def version(self):
    if self._finalized:
      return self._version
    with self._lock:
      return self._version

  @property
  def graph_def_versions(self):
    return self._graph_def_versions

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, seed):
    self._seed = seed

  @property
  def finalized(self):
    """True if this graph has been finalized."""
    return self._finalized

  def finalize(self):
    self._finalized = True

  def _unsafe_unfinalize(self):
    self._finalized = False

  def _get_control_flow_context(self):
    return self._control_flow_context

  def _set_control_flow_context(self, context):
    self._control_flow_context = context

  def _as_graph_def(self, from_version=None, add_shapes=False):
    outfile.write("in python/framework/ops Graph====_as_graph_def(self, from_version=None, add_shapes=False)======\n")
    with self._lock:
      graph = graph_pb2.GraphDef()
      graph.versions.CopyFrom(self._graph_def_versions)
      bytesize = 0
      for op_id in sorted(self._nodes_by_id):
        op = self._nodes_by_id[op_id]
        if from_version is None or op_id > from_version:
          graph.node.extend([op.node_def])
          if op.outputs and add_shapes:
            assert "_output_shapes" not in graph.node[-1].attr
            graph.node[-1].attr["_output_shapes"].list.shape.extend([
                output.get_shape().as_proto() for output in op.outputs])
          bytesize += op.node_def.ByteSize()
          if bytesize >= (1 << 31) or bytesize < 0:
            raise ValueError("GraphDef cannot be larger than 2GB.")
      if self._functions:
        for f in self._functions.values():
          bytesize += f.definition.ByteSize()
          if bytesize >= (1 << 31) or bytesize < 0:
            raise ValueError("GraphDef cannot be larger than 2GB.")
          graph.library.function.extend([f.definition])
          if f.grad_func_name:
            grad_def = function_pb2.GradientDef()
            grad_def.function_name = f.name
            grad_def.gradient_func = f.grad_func_name
            graph.library.gradient.extend([grad_def])
      return graph, self._version

  def as_graph_def(self, from_version=None, add_shapes=False):
    outfile.write("in python/framework/ops Graph====as_graph_def(self, from_version=None, add_shapes=False)======\n")
    result, _ = self._as_graph_def(from_version, add_shapes)
    return result

#被gradient_impl引用
  def _is_function(self, name):
    return name in self._functions

  #被_get_graph_from_inputs引用
  @property
  def building_function(self):
    return self._building_function

#被constant_op的constant引用
  def create_op(self, op_type, inputs, dtypes,input_types=None, name=None, attrs=None, op_def=None,compute_shapes=True, compute_device=True):
    outfile.write("in python/framework/ops Graph====create_op======\n")
    self._check_not_finalized()
    if name is None:name = op_type
    if name and name[-1] == "/":name = name[:-1]
    else:name = self.unique_name(name)
    node_def = _NodeDef(op_type, name, device=None, attrs=attrs)
    for key, value in self._attr_scope_map.items():
      if key not in node_def.attr:node_def.attr[key].CopyFrom(value)
    try:
      kernel_label = self._op_to_kernel_label_map[op_type]
      node_def.attr["_kernel"].CopyFrom(attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_label)))
    except KeyError:pass
    try:
      mapped_op_type = self._gradient_override_map[op_type]
      node_def.attr["_gradient_op_type"].CopyFrom(attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
    except KeyError:pass
    control_inputs = self._control_dependencies_for_inputs(inputs)
    ret = Operation(node_def, self, inputs=inputs, output_types=dtypes,control_inputs=control_inputs, input_types=input_types,original_op=self._default_original_op, op_def=op_def)
    if compute_shapes:set_shapes_for_outputs(ret)
    self._add_op(ret)
    self._record_op_seen_by_control_dependencies(ret)
    if compute_device:self._apply_device_functions(ret)
    if self._colocation_stack:
      all_colocation_groups = []
      for colocation_op in self._colocation_stack:
        all_colocation_groups.extend(colocation_op.colocation_groups())
        if colocation_op.device:
          if ret.device and ret.device != colocation_op.device:
            print("ret.device and ret.device != colocation_op.device")
          else:
            ret._set_device(colocation_op.device)
      all_colocation_groups = sorted(set(all_colocation_groups))
      ret.node_def.attr["_class"].CopyFrom(attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(s=all_colocation_groups)))
    if (self._container and op_type in self._registered_ops and self._registered_ops[op_type].is_stateful and "container" in ret.node_def.attr and not ret.node_def.attr["container"].s):
      ret.node_def.attr["container"].CopyFrom(attr_value_pb2.AttrValue(s=compat.as_bytes(self._container)))
    return ret

#被control_dependencies(self, control_inputs)引用
  def as_graph_element(self, obj, allow_tensor=True, allow_operation=True):
    outfile.write("in python/framework/ops Graph====as_graph_element======\n")
    if self._finalized:return self._as_graph_element_locked(obj, allow_tensor, allow_operation)
    with self._lock:return self._as_graph_element_locked(obj, allow_tensor, allow_operation)

  #被as_graph_element引用
  def _as_graph_element_locked(self, obj, allow_tensor, allow_operation):
    outfile.write("in python/framework/ops Graph====_as_graph_element_locked======\n")
    if allow_tensor and allow_operation:types_str = "Tensor or Operation"
    elif allow_tensor:types_str = "Tensor"
    elif allow_operation:types_str = "Operation"
    temp_obj = _as_graph_element(obj)
    if temp_obj is not None:obj = temp_obj
    if isinstance(obj, compat.bytes_or_text_types):
      name = compat.as_str(obj)
      if ":" in name and allow_tensor:
        op_name, out_n = name.split(":")
        out_n = int(out_n)
        if op_name in self._nodes_by_name:
          op = self._nodes_by_name[op_name]
        return op.outputs[out_n]
      elif ":" not in name and allow_operation:
        return self._nodes_by_name[name]
    elif isinstance(obj, Tensor) and allow_tensor:
      return obj
    elif isinstance(obj, Operation) and allow_operation:
      return obj

#被Operation的init引用
  def _next_id(self):
    outfile.write("in python/framework/ops Graph====_next_id======\n")
    self._check_not_finalized()
    with self._lock:
      self._next_id_counter += 1
      return self._next_id_counter


#被gradient_impl的_PendingCount引用
  @property
  def _last_id(self):
    outfile.write("in python/framework/ops Graph====_last_id======\n")
    return self._next_id_counter

#被session的__enter__引用
  def as_default(self):
    outfile.write("in python/framework/ops Graph====as_default======\n")
    return _default_graph_stack.get_controller(self)

#被add_to_collections(self, names, value)引用
  def add_to_collection(self, name, value):
    outfile.write("in ops=========add_to_collection=====")
    self._check_not_finalized()
    with self._lock:
      if name not in self._collections:
        self._collections[name] = [value]
      else:
        self._collections[name].append(value)

#被add_to_collections(names, value)引用
  def add_to_collections(self, names, value):
    outfile.write("in ops=========add_to_collections=====\n")
    names = (names,) if isinstance(names, six.string_types) else set(names)
    for name in names:
      self.add_to_collection(name, value)

#被get_collection_ref(key)引用
  def get_collection_ref(self, name):
    outfile.write("in ops=========get_collection_ref=====\n")
    with self._lock:
      coll_list = self._collections.get(name, None)
      if coll_list is None:
        coll_list = []
        self._collections[name] = coll_list
      return coll_list

#被get_collection(key, scope=None)引用
  def get_collection(self, name, scope=None):
    outfile.write("in ops=========get_collection=====\n")
    with self._lock:
      coll_list = self._collections.get(name, None)
      if coll_list is None:
        return []
      if scope is None:
        return list(coll_list)
      else:
        c = []
        regex = re.compile(scope)
        for item in coll_list:
          if hasattr(item, "name") and regex.match(item.name):
            c.append(item)
        return c

#framework的meta_graph引用
  def get_all_collection_keys(self):
    outfile.write("in ops=========get_all_collection_keys=====\n")
    with self._lock:
      return [x for x in self._collections if isinstance(x, six.string_types)]

#被gradients_impl引用
  @contextlib.contextmanager
  def _original_op(self, op):
    outfile.write("in ops=========_original_op=====\n")
    old_original_op = self._default_original_op
    try:
      self._default_original_op = op
      yield
    finally:
      self._default_original_op = old_original_op

#被name_scope(name, default_name=None, values=None)引用
  @contextlib.contextmanager
  def name_scope(self, name):
    outfile.write("in python/framework/ops name_scope(self, name)\n")
    if name:
      if self._name_stack:
        if not _VALID_SCOPE_NAME_REGEX.match(name):
          raise ValueError("'%s' is not a valid scope name" % name)
      else:
        if not _VALID_OP_NAME_REGEX.match(name):
          raise ValueError("'%s' is not a valid scope name" % name)
    try:
      old_stack = self._name_stack
      if not name:
        new_stack = None
      elif name and name[-1] == "/":
        new_stack = name[:-1]
      else:
        new_stack = self.unique_name(name)
      self._name_stack = new_stack
      yield "" if new_stack is None else new_stack + "/"
    finally:
      self._name_stack = old_stack

#被name_scope(self, name)引用
  def unique_name(self, name, mark_as_used=True):
    outfile.write("in python/framework/ops unique_name\n")
    if self._name_stack:
      name = self._name_stack + "/" + name
    i = self._names_in_use.get(name, 0)
    if mark_as_used:
      self._names_in_use[name] = i + 1
    if i > 0:
      base_name = name
      while name in self._names_in_use:
        name = "%s_%d" % (base_name, i)
        i += 1
      if mark_as_used:
        self._names_in_use[name] = 1
    return name

#被colocate_with(op, ignore_existing=False)引用
  @contextlib.contextmanager
  def colocate_with(self, op, ignore_existing=False):
    outfile.write("in in python/framework/ops========colocate_with============\n")
    if op is None:
      raise ValueError("Tried to colocate with None")
    if not isinstance(op, Operation):
      op = convert_to_tensor_or_indexed_slices(op, as_ref=True).op
    device_fn_tmp = self._device_function_stack
    self._device_function_stack = []
    if ignore_existing:
      current_stack = self._colocation_stack
      self._colocation_stack = []
    self._colocation_stack.append(op)
    try:
      yield
    finally:
      self._device_function_stack = device_fn_tmp
      self._colocation_stack.pop()
      if ignore_existing:
        self._colocation_stack = current_stack

#被device引用
  @contextlib.contextmanager
  def device(self, device_name_or_function):
    outfile.write("in in python/framework/ops========device============\n")
    if (device_name_or_function is not None
        and not callable(device_name_or_function)):
      device_function = pydev.merge_device(device_name_or_function)
    else:
      device_function = device_name_or_function
    try:
      self._device_function_stack.append(device_function)
      yield
    finally:
      self._device_function_stack.pop()

#被create_op引用
  def _apply_device_functions(self, op):
    outfile.write("in in python/framework/ops========_apply_device_functions============\n")
    for device_function in reversed(self._device_function_stack):
      if device_function is None:
        break
      op._set_device(device_function(op))



#被control_dependencies(self, control_inputs)引用
  class _ControlDependenciesController(object):
    def __init__(self, graph, control_inputs):
      outfile.write("in in python/framework/ops========_ControlDependenciesController======__init__======\n")
      self._graph = graph
      if control_inputs is None:
        self._control_inputs = []
        self._new_stack = True
      else:
        self._control_inputs = control_inputs
        self._new_stack = False
      self._seen_nodes = set()
      self._old_stack = None
      self._old_control_flow_context = None
  #被variables的init_from_args引用
    def __enter__(self):
      outfile.write("in in python/framework/ops========_ControlDependenciesController======__enter__======\n")
      if self._new_stack:
        self._old_stack = self._graph._control_dependencies_stack
        self._graph._control_dependencies_stack = []
        self._old_control_flow_context = self._graph._get_control_flow_context()
        self._graph._set_control_flow_context(None)
      self._graph._push_control_dependencies_controller(self)
    def __exit__(self, unused_type, unused_value, unused_traceback):
      self._graph._pop_control_dependencies_controller(self)
      if self._new_stack:
        self._graph._control_dependencies_stack = self._old_stack
        self._graph._set_control_flow_context(self._old_control_flow_context)
    @property
    def control_inputs(self):
      return self._control_inputs
    def add_op(self, op):
      self._seen_nodes.add(op)
    def op_in_group(self, op):
      return op in self._seen_nodes
  def _push_control_dependencies_controller(self, controller):
    self._control_dependencies_stack.append(controller)
  def _pop_control_dependencies_controller(self, controller):
    assert self._control_dependencies_stack[-1] is controller
    self._control_dependencies_stack.pop()
  def _current_control_dependencies(self):
    ret = set()
    for controller in self._control_dependencies_stack:
      for op in controller.control_inputs:
        ret.add(op)
    return ret

 #被create_op引用
  def _control_dependencies_for_inputs(self, input_tensors):
    outfile.write("in in python/framework/ops========_control_dependencies_for_inputs======__enter__======\n")
    ret = []
    input_ops = set([t.op for t in input_tensors])
    for controller in self._control_dependencies_stack:
      dominated = False
      for op in input_ops:
        if controller.op_in_group(op):
          dominated = True
          break
      if not dominated:
        ret.extend([c for c in controller.control_inputs if c not in input_ops])
    return ret

  def _record_op_seen_by_control_dependencies(self, op):
    for controller in self._control_dependencies_stack:
      controller.add_op(op)

#被control_dependencies(control_inputs)引用
  def control_dependencies(self, control_inputs):
    outfile.write("in python/framework/ops ======control_dependencies(self, control_inputs)\n")
    if control_inputs is None:
      return self._ControlDependenciesController(self, None)
    control_ops = []
    current = self._current_control_dependencies()
    for c in control_inputs:
      c = self.as_graph_element(c)
      if isinstance(c, Tensor):
        c = c.op
      elif not isinstance(c, Operation):
        raise TypeError("Control input must be Operation or Tensor: %s" % c)
      if c not in current:
        control_ops.append(c)
        current.add(c)
    return self._ControlDependenciesController(self, control_ops)



  def prevent_feeding(self, tensor):
    self._unfeedable_tensors.add(tensor)

def device(device_name_or_function):
  return get_default_graph().device(device_name_or_function)


#被op_def_lib的_MaybeColocateWith引用
def colocate_with(op, ignore_existing=False):
  outfile.write("in python/framework/ops ======colocate_with(op, ignore_existing=False)\n")
  return get_default_graph().colocate_with(op, ignore_existing)


#被python/ops/variables引用
def control_dependencies(control_inputs):
  outfile.write("in python/framework/ops ======control_dependencies(control_inputs)\n")
  return get_default_graph().control_dependencies(control_inputs)


class _DefaultStack(threading.local):
  def __init__(self):
    super(_DefaultStack, self).__init__()
    self._enforce_nesting = True
    self.stack = []

  def get_default(self):
    return self.stack[-1] if len(self.stack) >= 1 else None

  def reset(self):
    self.stack = []

  @property
  def enforce_nesting(self):
    return self._enforce_nesting

  @enforce_nesting.setter
  def enforce_nesting(self, value):
    self._enforce_nesting = value

#被default_session引用
  @contextlib.contextmanager
  def get_controller(self, default):
    try:
      self.stack.append(default)
      yield default
    finally:
      if self._enforce_nesting:
        if self.stack[-1] is not default:
          raise AssertionError(
              "Nesting violated for default stack of %s objects"
              % type(default))
        self.stack.pop()
      else:
        self.stack.remove(default)

_default_session_stack = _DefaultStack()


def default_session(session):
  return _default_session_stack.get_controller(session)


def get_default_session():
  return _default_session_stack.get_default()


def _eval_using_default_session(tensors, feed_dict, graph, session=None):
  outfile.write("in python/framework/ops ======_eval_using_default_session\n")
  if session is None:
    session = get_default_session()
  return session.run(tensors, feed_dict)


class _DefaultGraphStack(_DefaultStack):
  def __init__(self):
    super(_DefaultGraphStack, self).__init__()
    outfile.write("in python/framework/ops =====_DefaultGraphStack====__init__=================================\n")
    self._global_default_graph = None

  def get_default(self,outfile):
    outfile.write("in python/framework/ops==========get_default=================================\n")
    ret = super(_DefaultGraphStack, self).get_default()
    if ret is None:
      ret = self._GetGlobalDefaultGraph(outfile)
    return ret

  def _GetGlobalDefaultGraph(self,outfile):
    outfile.write("in python/framework/ops==========class _DefaultGraphStack====_GetGlobalDefaultGraph=================================\n")
    if self._global_default_graph is None:
      self._global_default_graph = Graph()
    return self._global_default_graph

  def reset(self):
    super(_DefaultGraphStack, self).reset()
    self._global_default_graph = None

_default_graph_stack = _DefaultGraphStack()

def reset_default_graph():
  _default_graph_stack.reset()

def get_default_graph():
  outfile.write("in python/framework/ops==========get_default_graph=================================\n")
  return _default_graph_stack.get_default(outfile)

def _get_graph_from_inputs(op_input_list, graph=None):
  outfile.write("in gen_array_ops==========_get_graph_from_inputs=================================\n")
  if get_default_graph().building_function:
    return get_default_graph()
  op_input_list = tuple(op_input_list)  # Handle generators correctly
  original_graph_element = None
  for op_input in op_input_list:
    graph_element = None
    if isinstance(op_input, (Operation, _TensorLike)):
      graph_element = op_input
    else:
      graph_element = _as_graph_element(op_input)
    if graph_element is not None:
      if not graph:
        original_graph_element = graph_element
        graph = graph_element.graph
  return graph or get_default_graph()

#被control_flow_ops的ops.register_proto_function引用
class GraphKeys(object):
  GLOBAL_VARIABLES = "variables"
  LOCAL_VARIABLES = "local_variables"
  MODEL_VARIABLES = "model_variables"
  TRAINABLE_VARIABLES = "trainable_variables"
  SUMMARIES = "summaries"
  QUEUE_RUNNERS = "queue_runners"
  TABLE_INITIALIZERS = "table_initializer"
  ASSET_FILEPATHS = "asset_filepaths"
  MOVING_AVERAGE_VARIABLES = "moving_average_variables"
  REGULARIZATION_LOSSES = "regularization_losses"
  CONCATENATED_VARIABLES = "concatenated_variables"
  SAVERS = "savers"
  WEIGHTS = "weights"
  BIASES = "biases"
  ACTIVATIONS = "activations"
  UPDATE_OPS = "update_ops"
  LOSSES = "losses"
  SAVEABLE_OBJECTS = "saveable_objects"
  RESOURCES = "resources"
  LOCAL_RESOURCES = "local_resources"
  TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables"
  INIT_OP = "init_op"
  LOCAL_INIT_OP = "local_init_op"
  READY_OP = "ready_op"
  READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op"
  SUMMARY_OP = "summary_op"
  GLOBAL_STEP = "global_step"
  EVAL_STEP = "eval_step"
  TRAIN_OP = "train_op"
  COND_CONTEXT = "cond_context"
  WHILE_CONTEXT = "while_context"

  @decorator_utils.classproperty
  def VARIABLES(cls):  # pylint: disable=no-self-argument
    logging.warning("VARIABLES collection name is deprecated, "
                    "please use GLOBAL_VARIABLES instead; "
                    "VARIABLES will be removed after 2017-03-02.")
    return cls.GLOBAL_VARIABLES


def add_to_collection(name, value):
  outfile.write("in python/framework/ops ========add_to_collection===============\n")
  get_default_graph().add_to_collection(name, value)


def add_to_collections(names, value):
  outfile.write("in python/framework/ops ========add_to_collections(names, value)===============\n")
  get_default_graph().add_to_collections(names, value)


def get_collection_ref(key):
  outfile.write("in python/framework/ops ========get_collection_ref(key)===============\n")
  return get_default_graph().get_collection_ref(key)


def get_collection(key, scope=None):
  outfile.write("in python/framework/ops ========get_collection(key, scope=None)===============\n")
  return get_default_graph().get_collection(key, scope)


def get_all_collection_keys():
  outfile.write("in python/framework/ops ========get_all_collection_keys()===============\n")
  return get_default_graph().get_all_collection_keys()


@contextlib.contextmanager
def name_scope(name, default_name=None, values=None):
  outfile.write("in python/framework/ops =====namescope\n")
  n = default_name if name is None else name
  if values is None:
    values = []
  g = _get_graph_from_inputs(values)
  with g.as_default(), g.name_scope(n) as scope:
    yield scope

def strip_name_scope(name, export_scope):
  if export_scope:
    str_to_replace = r"([\^]|loc:@|^)" + export_scope + r"[\/]+(.*)"
    return re.sub(str_to_replace, r"\1\2", compat.as_str(name), count=1)
  else:
    return name



@contextlib.contextmanager
def op_scope(values, name, default_name=None):
  with name_scope(name, default_name=default_name, values=values) as scope:
    yield scope


_proto_function_registry = registry.Registry("proto functions")


def register_proto_function(collection_name, proto_type=None, to_proto=None,from_proto=None):
  outfile.write("in python/framework/ops = register_proto_function=====================\n")
  if to_proto and not callable(to_proto):
    raise TypeError("to_proto must be callable.")
  if from_proto and not callable(from_proto):
    raise TypeError("from_proto must be callable.")

  _proto_function_registry.register((proto_type, to_proto, from_proto),
                                    collection_name)




def _operation_conversion_error(op, dtype=None, name=None, as_ref=False):
  """Produce a nice error if someone converts an Operation to a Tensor."""
  raise TypeError(
      ("Can't convert Operation '%s' to Tensor "
       "(target dtype=%r, name=%r, as_ref=%r)") %
      (op.name, dtype, name, as_ref))


register_tensor_conversion_function(Operation, _operation_conversion_error)
