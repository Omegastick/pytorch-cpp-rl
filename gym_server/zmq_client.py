"""
Pytorch-cpp-rl OpenAI gym server ZMQ client.
"""
import logging
import zmq
import msgpack

from gym_server.messages import Message


class ZmqClient:
    """
    Class for connecting to and communicating with pytorch-cpp-rl.
    """

    def __init__(self, url: str = 'tcp://127.0.0.1:10201'):
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        logging.info("Waiting for connection to client")
        self.socket.connect(url)
        self.socket.send_string("Establishing connection...")
        logging.info(self.socket.recv_string())

    def receive(self) -> dict:
        """
        Waits for a message from pytorch-cpp-rl.
        Returns the received message obejct in dictionary form.
        """
        msg = self.socket.recv()
        response = msgpack.unpackb(msg, raw=False)
        return response

    def send(self, message: Message):
        """
        Sends a message to the server.
        """
        self.socket.send(message.to_msg())
