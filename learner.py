import importlib
import argparse
import io
import socket
import threading
import grpc
from concurrent import futures
import time
import torch
import logging
from protos import learner_pb2, learner_pb2_grpc
from protos import leader_pb2, leader_pb2_grpc
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GRPC_STUB_OPTIONS = [
    ('grpc.max_send_message_length', 50 * 1024 * 1024),  # For example, 50 MB
    ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # For example, 50 MB
]

class LearnerService(learner_pb2_grpc.LearnerServiceServicer):
    def __init__(self, network_addr, leader_stub, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = None
        self.model = None
        self.criterion = None
        self.leader_stub = leader_stub
        self.network_addr = network_addr
        self.data_batches = []
        self.sync_model_event = threading.Event()

    def load_model(self):
        logging.info("Getting Model...")
        model_stream = self.leader_stub.GetModel(leader_pb2.Empty())


        model_dir = './learner_model_artifacts'
        model_path = os.path.join(model_dir, 'model.py')

        os.makedirs(model_dir, exist_ok=True)

        with open(model_path, 'wb') as model_file:
            for model_data in model_stream:
                model_file.write(model_data.chunk)

            
            # dynamically import the model
            spec = importlib.util.spec_from_file_location('model_module', model_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
                # load relevant variables from the file
            self.device = model_module.device
            self.model = model_module.model.to(self.device)
            self.criterion = model_module.criterion

            logging.info('Model loaded successfully and moved to device')

            