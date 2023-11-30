import numpy as np
import cv2 as cv
from tqdm import tqdm
import time

from client_func import caption_video_batch

CAPTIONS_MODEL_URL = 'http://localhost:12345/predictions/blip2'   
n_batch_frames = 50

print(f'captions_model_url={CAPTIONS_MODEL_URL}')
print(f'n_batch_frames={n_batch_frames}')

def caption_video(video_path):
    
    cap = cv.VideoCapture(video_path)

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    info = f'{video_path} with {total_frames} frames ({frame_width}x{frame_height})'
    
    print(info)

    i_frame = 0
    captions = []
    
    with tqdm(desc=info,
              total=total_frames,
              initial=i_frame,
              unit_divisor=1,
              miniters=n_batch_frames) as bar:
        while cap.isOpened():
            frames = None
            for _ in range(n_batch_frames):
                ret, frame = cap.read()
                if not ret:
                    # skip blank frames
                    continue
                else:
                    i_frame += 1 
                    bar.update(1)

                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = np.expand_dims(frame,0)
                frames = frame if frames is None else np.concatenate((frames,frame))

            if isinstance(frames,np.ndarray):
                caps = caption_video_batch(url=CAPTIONS_MODEL_URL,frames=frames)
                captions.extend(caps)
            else:
                return captions

video_path = '../sample_video/test.mp4'

start_time = time.time()
captions = caption_video(video_path = video_path)
print("--- %s seconds ---" % (time.time() - start_time))
print(f'len(captions)={len(captions)}')
print('excerpt:')
for c in captions[::5][:10]:
    print(c.strip('\n'))

