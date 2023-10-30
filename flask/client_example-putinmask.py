import requests
import numpy as np
import pickle
import cv2 as cv
from time import sleep
import time
from tqdm import tqdm

CAPTIONS_MODEL_URL = 'http://localhost:5000/caption'
n_batch_frames = 100

print(f'captions_model_url={CAPTIONS_MODEL_URL}')
print(f'n_batch_frames={n_batch_frames}')

def captions_to_putinmask(captions):
    return ['putin' in x.lower() for x in captions]
def encode_putinmask(putinmask):
    return ''.join([str(x*1) for x in putinmask])
def decode_putinmask(encoded_putinmask):
    return [bool(int(x)) for x in encoded_putinmask]

def create_putinmask(video_path):
    
    cap = cv.VideoCapture(video_path)

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    info = f'{video_path} with {total_frames} frames ({frame_width}x{frame_height})'

    print(info)
    
    i_frame = 0
    putinmask = []
    
    with tqdm(desc=info,
              total=total_frames,
              initial=i_frame,
              # unit_scale=True,
              unit_divisor=1,
              miniters=n_batch_frames) as bar:
        while cap.isOpened():

            # resuming from start_frame  
            frames = None
            for _ in range(n_batch_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    i_frame += 1 
                    bar.update(1)

                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = np.expand_dims(frame,0)
                frames = frame if frames is None else np.concatenate((frames,frame))

            if isinstance(frames,np.ndarray):
                files = {'media': pickle.dumps(frames)}
                while True:
                    try:
                        response = requests.post(CAPTIONS_MODEL_URL, files=files)
                        if response.status_code == 200:
                            break
                    except:
                        print(f'problem {response.status_code}')
                        sleep(5)
                        pass

                captions = response.json()
                out = captions_to_putinmask(captions)
                encoded_putinmask = encode_putinmask(out)
                putinmask.extend(encoded_putinmask)
            else:
                return putinmask

video_path = '../sample_video/putin_test.mp4'
start_time = time.time()
putinmask = create_putinmask(video_path = video_path)
print("--- %s seconds ---" % (time.time() - start_time))
print(f'len(putinmask)={len(putinmask)}')
print('putinmask:')
print(''.join(putinmask))