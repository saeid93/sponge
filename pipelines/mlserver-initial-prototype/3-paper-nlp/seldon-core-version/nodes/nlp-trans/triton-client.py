import argparse
from functools import partial
import os
import sys
from io import BytesIO

import numpy as np

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
from datasets import load_dataset


gateway_endpoint="localhost:32000"
deployment_name = 'nlp'
namespace = "default"
endpoint = f"{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"
print(endpoint)
data=["""
Après des décennies en tant que pratiquant d'arts martiaux et coureur, Wes a "trouvé" le yoga en 2010.
Il en est venu à apprécier que son ampleur et sa profondeur fournissent un merveilleux lest pour stabiliser
le corps et l'esprit dans le style de vie rapide et axé sur la technologie d'aujourd'hui ;
le yoga est un antidote au stress et une voie vers une meilleure compréhension de soi et des autres.
Il est instructeur de yoga certifié RYT 500 du programme YogaWorks et s'est formé avec des maîtres contemporains,
dont Mme Maty Ezraty, co-fondatrice de YogaWorks et maître instructeur des traditions Iyengar et Ashtanga,
ainsi qu'une spécialisation avec M. Bernie. Clark, un maître instructeur de la tradition Yin.
Ses cours reflètent ces traditions, où il combine la base fondamentale d'un alignement précis avec des éléments
d'équilibre et de concentration. Ceux-ci s'entremêlent pour aider à fournir une voie pour cultiver une conscience
de vous-même, des autres et du monde qui vous entoure, ainsi que pour créer un refuge contre le style de vie rapide
et axé sur la technologie d'aujourd'hui. Il enseigne à aider les autres à réaliser le même bénéfice de la pratique dont il a lui-même bénéficié.
Mieux encore, les cours de yoga sont tout simplement merveilleux :
ils sont à quelques instants des exigences de la vie où vous pouvez simplement prendre soin de vous physiquement et émotionnellement.
    """]
input_name = "text_inputs"
shape = [1]
shape_n = np.array(shape)
dtype = "BYTES"
inputs = [httpclient.InferInput(input_name, shape, dtype)]

data_bytes = data[0].encode('utf-8')
in0n = np.array([str(x).encode('utf-8') for x in data],
                    dtype=np.object_)
input0_data = in0n.reshape(shape_n.shape)
inputs[0].set_data_from_numpy(input0_data)

try:
    triton_client = httpclient.InferenceServerClient(
                url=endpoint, verbose=True, concurrency=1)
except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

# try:
triton_client.infer(
                    'nlp-trans',
                    inputs,

                        )

# except Exception as e:
#             print("inference failed: " + str(e))
#             sys.exit(1)