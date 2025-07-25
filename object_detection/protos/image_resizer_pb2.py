# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: image_resizer.proto
# Protobuf Python Version: 5.27.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    1,
    '',
    'image_resizer.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13image_resizer.proto\x12\x17object_detection.protos\"\xb5\x03\n\x0cImageResizer\x12T\n\x19keep_aspect_ratio_resizer\x18\x01 \x01(\x0b\x32/.object_detection.protos.KeepAspectRatioResizerH\x00\x12I\n\x13\x66ixed_shape_resizer\x18\x02 \x01(\x0b\x32*.object_detection.protos.FixedShapeResizerH\x00\x12\x44\n\x10identity_resizer\x18\x03 \x01(\x0b\x32(.object_detection.protos.IdentityResizerH\x00\x12U\n\x19\x63onditional_shape_resizer\x18\x04 \x01(\x0b\x32\x30.object_detection.protos.ConditionalShapeResizerH\x00\x12P\n\x17pad_to_multiple_resizer\x18\x05 \x01(\x0b\x32-.object_detection.protos.PadToMultipleResizerH\x00\x42\x15\n\x13image_resizer_oneof\"\x11\n\x0fIdentityResizer\"\x80\x02\n\x16KeepAspectRatioResizer\x12\x1a\n\rmin_dimension\x18\x01 \x01(\x05:\x03\x36\x30\x30\x12\x1b\n\rmax_dimension\x18\x02 \x01(\x05:\x04\x31\x30\x32\x34\x12\x44\n\rresize_method\x18\x03 \x01(\x0e\x32#.object_detection.protos.ResizeType:\x08\x42ILINEAR\x12#\n\x14pad_to_max_dimension\x18\x04 \x01(\x08:\x05\x66\x61lse\x12#\n\x14\x63onvert_to_grayscale\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\x1d\n\x15per_channel_pad_value\x18\x06 \x03(\x02\"\xa7\x01\n\x11\x46ixedShapeResizer\x12\x13\n\x06height\x18\x01 \x01(\x05:\x03\x33\x30\x30\x12\x12\n\x05width\x18\x02 \x01(\x05:\x03\x33\x30\x30\x12\x44\n\rresize_method\x18\x03 \x01(\x0e\x32#.object_detection.protos.ResizeType:\x08\x42ILINEAR\x12#\n\x14\x63onvert_to_grayscale\x18\x04 \x01(\x08:\x05\x66\x61lse\"\xb9\x02\n\x17\x43onditionalShapeResizer\x12\\\n\tcondition\x18\x01 \x01(\x0e\x32@.object_detection.protos.ConditionalShapeResizer.ResizeCondition:\x07GREATER\x12\x1b\n\x0esize_threshold\x18\x02 \x01(\x05:\x03\x33\x30\x30\x12\x44\n\rresize_method\x18\x03 \x01(\x0e\x32#.object_detection.protos.ResizeType:\x08\x42ILINEAR\x12#\n\x14\x63onvert_to_grayscale\x18\x04 \x01(\x08:\x05\x66\x61lse\"8\n\x0fResizeCondition\x12\x0b\n\x07INVALID\x10\x00\x12\x0b\n\x07GREATER\x10\x01\x12\x0b\n\x07SMALLER\x10\x02\"P\n\x14PadToMultipleResizer\x12\x13\n\x08multiple\x18\x01 \x01(\x05:\x01\x31\x12#\n\x14\x63onvert_to_grayscale\x18\x04 \x01(\x08:\x05\x66\x61lse*G\n\nResizeType\x12\x0c\n\x08\x42ILINEAR\x10\x00\x12\x14\n\x10NEAREST_NEIGHBOR\x10\x01\x12\x0b\n\x07\x42ICUBIC\x10\x02\x12\x08\n\x04\x41REA\x10\x03')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'image_resizer_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_RESIZETYPE']._serialized_start=1334
  _globals['_RESIZETYPE']._serialized_end=1405
  _globals['_IMAGERESIZER']._serialized_start=49
  _globals['_IMAGERESIZER']._serialized_end=486
  _globals['_IDENTITYRESIZER']._serialized_start=488
  _globals['_IDENTITYRESIZER']._serialized_end=505
  _globals['_KEEPASPECTRATIORESIZER']._serialized_start=508
  _globals['_KEEPASPECTRATIORESIZER']._serialized_end=764
  _globals['_FIXEDSHAPERESIZER']._serialized_start=767
  _globals['_FIXEDSHAPERESIZER']._serialized_end=934
  _globals['_CONDITIONALSHAPERESIZER']._serialized_start=937
  _globals['_CONDITIONALSHAPERESIZER']._serialized_end=1250
  _globals['_CONDITIONALSHAPERESIZER_RESIZECONDITION']._serialized_start=1194
  _globals['_CONDITIONALSHAPERESIZER_RESIZECONDITION']._serialized_end=1250
  _globals['_PADTOMULTIPLERESIZER']._serialized_start=1252
  _globals['_PADTOMULTIPLERESIZER']._serialized_end=1332
# @@protoc_insertion_point(module_scope)
