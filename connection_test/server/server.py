import os
import grpc
import chunk_pb2
import chunk_pb2_grpc
from concurrent import futures
import time

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

'''Chunk size for shrinking the input to smaller sizes for transmitting'''
chunk_size = 1024 * 1024  # 1MB


'''
    Saving the downloaded image to a file
'''


def save_batch(batch):
    for ch in batch.chunks:
        filename = ch.name
        print(str(ch.identification_number) + filename)
        with open(str(ch.identification_number) + filename, 'wb') as f:
            f.write(ch.buffer)


class FileServer(chunk_pb2_grpc.FileServerServicer):
    def upload_batch(self, request_iterator, context):
        save_batch(request_iterator)
        return chunk_pb2.Reply(result='1')

def run(server_port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chunk_pb2_grpc.add_FileServerServicer_to_server(FileServer(), server)
    server.add_insecure_port('[::]:'+server_port)
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    run('8008')
