# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: learner.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rlearner.proto\x12\x07learner\"\x1b\n\nModelState\x12\r\n\x05\x63hunk\x18\x01 \x01(\x0c\"\x07\n\x05\x45mpty\"\'\n\x03\x41\x63k\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t2x\n\x0eLearnerService\x12/\n\rStartTraining\x12\x0e.learner.Empty\x1a\x0c.learner.Ack\"\x00\x12\x35\n\x0eSyncModelState\x12\x13.learner.ModelState\x1a\x0c.learner.Ack\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'learner_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_MODELSTATE']._serialized_start=26
  _globals['_MODELSTATE']._serialized_end=53
  _globals['_EMPTY']._serialized_start=55
  _globals['_EMPTY']._serialized_end=62
  _globals['_ACK']._serialized_start=64
  _globals['_ACK']._serialized_end=103
  _globals['_LEARNERSERVICE']._serialized_start=105
  _globals['_LEARNERSERVICE']._serialized_end=225
# @@protoc_insertion_point(module_scope)
