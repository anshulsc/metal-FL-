# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import leader_pb2 as leader__pb2


class LeaderServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RegisterLearner = channel.unary_unary(
                '/leader.LeaderService/RegisterLearner',
                request_serializer=leader__pb2.LearnerInfo.SerializeToString,
                response_deserializer=leader__pb2.AckWithMetadata.FromString,
                )
        self.GetModel = channel.unary_stream(
                '/leader.LeaderService/GetModel',
                request_serializer=leader__pb2.Empty.SerializeToString,
                response_deserializer=leader__pb2.ModelChunk.FromString,
                )
        self.GetData = channel.unary_stream(
                '/leader.LeaderService/GetData',
                request_serializer=leader__pb2.LearnerDataRequest.SerializeToString,
                response_deserializer=leader__pb2.DataChunk.FromString,
                )
        self.AccumulateGradients = channel.unary_unary(
                '/leader.LeaderService/AccumulateGradients',
                request_serializer=leader__pb2.GradientData.SerializeToString,
                response_deserializer=leader__pb2.Ack.FromString,
                )


class LeaderServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RegisterLearner(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def AccumulateGradients(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LeaderServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RegisterLearner': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterLearner,
                    request_deserializer=leader__pb2.LearnerInfo.FromString,
                    response_serializer=leader__pb2.AckWithMetadata.SerializeToString,
            ),
            'GetModel': grpc.unary_stream_rpc_method_handler(
                    servicer.GetModel,
                    request_deserializer=leader__pb2.Empty.FromString,
                    response_serializer=leader__pb2.ModelChunk.SerializeToString,
            ),
            'GetData': grpc.unary_stream_rpc_method_handler(
                    servicer.GetData,
                    request_deserializer=leader__pb2.LearnerDataRequest.FromString,
                    response_serializer=leader__pb2.DataChunk.SerializeToString,
            ),
            'AccumulateGradients': grpc.unary_unary_rpc_method_handler(
                    servicer.AccumulateGradients,
                    request_deserializer=leader__pb2.GradientData.FromString,
                    response_serializer=leader__pb2.Ack.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'leader.LeaderService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LeaderService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RegisterLearner(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/leader.LeaderService/RegisterLearner',
            leader__pb2.LearnerInfo.SerializeToString,
            leader__pb2.AckWithMetadata.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/leader.LeaderService/GetModel',
            leader__pb2.Empty.SerializeToString,
            leader__pb2.ModelChunk.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/leader.LeaderService/GetData',
            leader__pb2.LearnerDataRequest.SerializeToString,
            leader__pb2.DataChunk.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def AccumulateGradients(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/leader.LeaderService/AccumulateGradients',
            leader__pb2.GradientData.SerializeToString,
            leader__pb2.Ack.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
