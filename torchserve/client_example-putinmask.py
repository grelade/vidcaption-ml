import os
import numpy as np
import pickle
import cv2 as cv
from time import sleep
from tqdm import tqdm
import time

from client_func import caption_video_stream

CAPTIONS_MODEL_URL = 'http://localhost:12345/predictions/blip2'

print(f'captions_model_url={CAPTIONS_MODEL_URL}')

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
    
    captions = caption_video_stream(CAPTIONS_MODEL_URL, cap)
    putinmask = captions_to_putinmask(captions)
    encoded_putinmask = encode_putinmask(putinmask)
    
    return encoded_putinmask

video_path = '../sample_video/putin_test.mp4'
video_path = '../../videos/41d53b862a9be337892a.mp4'

start_time = time.time()
putinmask = create_putinmask(video_path = video_path)
print("--- %s seconds ---" % (time.time() - start_time))
print(f'len(putinmask)={len(putinmask)}')
print('putinmask:')
print(''.join(putinmask))

