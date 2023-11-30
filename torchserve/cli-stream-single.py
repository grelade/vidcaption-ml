import cv2 as cv
import time

from client_func import caption_video_stream

CAPTIONS_MODEL_URL = 'http://localhost:12345/predictions/blip2'

print(f'captions_model_url={CAPTIONS_MODEL_URL}')

def caption_video(video_path):
    
    cap = cv.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    info = f'{video_path} with {total_frames} frames ({frame_width}x{frame_height})'
    
    print(info)
    
    start_time = time.time()
    captions = caption_video_stream(url = CAPTIONS_MODEL_URL, cap = cap)
    
    return captions
    
video_path = '../sample_video/test.mp4'

start_time = time.time()
captions = caption_video(video_path = video_path)
print("--- %s seconds ---" % (time.time() - start_time))
print(f'len(captions)={len(captions)}')
print('excerpt:')
for c in captions[::5][:10]:
    print(c.strip('\n'))

