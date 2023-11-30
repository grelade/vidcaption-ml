import numpy as np
import time

from client_func import caption_video_batch

CAPTIONS_MODEL_URL = 'http://localhost:12345/predictions/blip2'
n_batch_frames = 10

print(f'captions_model_url={CAPTIONS_MODEL_URL}')
print(f'n_batch_frames={n_batch_frames}')

frames = (np.random.rand(n_batch_frames,3,1280,760)*255).astype(np.uint8)

print(f'frames.shape={frames.shape}')

start_time = time.time()
captions = caption_video_batch(url=CAPTIONS_MODEL_URL,frames=frames)
print("--- %s seconds ---" % (time.time() - start_time))
print(f'len(captions)={len(captions)}')
print('excerpt:')
for c in captions:
    print(c.strip('\n'))