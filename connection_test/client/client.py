import os
import grpc
import chunk_pb2
import chunk_pb2_grpc
import datetime

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

'''Chunk size for shrinking the input to smaller sizes for transmitting'''
chunk_size = 1024 * 1024  # 1MB

'''
    Shrinking the image and making it ready for transmitting
'''


def get_file_chunks(filename, query_path, sla, importance, original_path):
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(chunk_size)
            if len(piece) == 0:
                return
            yield chunk_pb2.Chunk(buffer=piece, name=filename, query_path=query_path, sla=sla, importance=importance,
                                  spent_time='', original_path=original_path)


'''
    Saving the downloaded image to a file
'''


def save_chunks_to_file(chunks, filename):
    with open(filename, 'wb') as f:
        for chunk in chunks:
            f.write(chunk.buffer)


'''
The client class for transmitting data
'''
# {
#     metadata: "ddd"
#     data: "byte"
# }

class FileClient:
    def __init__(self, address):
        channel = grpc.insecure_channel(address)
        self.stub = chunk_pb2_grpc.FileServerStub(channel)

    def upload_batch(self, filename, ide):
        batch = chunk_pb2.BatchChunk()
        for i in range(len(filename)):
            msg = batch.chunks.add()
            msg.identification_number = ide[i]
            msg.buffer = open(filename[i], 'rb').read()
            msg.name = filename[i]
        response = self.stub.upload_batch(batch)
        assert response.result == '1'

'''
Testing that the gRPC is working between a client and a server
'''

if __name__ == '__main__':
    print(datetime.datetime.now())
    server_address = '130.83.163.232:8008'
    file_name = 'dog.jpg'
    id_num = 1
    client = FileClient(server_address)
    client.upload_batch([file_name], [id_num])
    print(datetime.datetime.now())
