import numpy as np
import requests
import pickle
import time

CAPTIONS_MODEL_URL = 'http://localhost:5000/caption'
n_batch_frames = 10

print(f'captions_model_url={CAPTIONS_MODEL_URL}')
print(f'n_batch_frames={n_batch_frames}')

frames = (np.random.rand(n_batch_frames,3,1280,760)*255).astype(np.uint8)
files = {'media': pickle.dumps(frames)}

print(f'frames.shape={frames.shape}')

start_time = time.time()
response = requests.post(CAPTIONS_MODEL_URL, files=files)
captions = response.json()
print("--- %s seconds ---" % (time.time() - start_time))
print(f'len(captions)={len(captions)}')
print('excerpt:')
for c in captions:
    print(c.strip('\n'))