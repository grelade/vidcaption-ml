import asyncio
import aiohttp
import cv2 as cv
import pickle
import time
import numpy as np

CAPTIONS_MODEL_URL = 'http://localhost:12345/predictions/blip2'
PREBATCH_SIZE = 8
N_WORKERS = 10

async def caption_video(video_path: str) -> bool:
    
    vframe_queue = asyncio.PriorityQueue(maxsize=10)
    caption_queue = asyncio.PriorityQueue()
    
    async def video_reader():
        '''
        reads frames from video_path and puts them on vframe_queue in batches of size PREBATCH_SIZE
        '''
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        frames = []
        i_batch = 0
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret: 
                break
            else:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frames.append(frame)
                if len(frames) == PREBATCH_SIZE:
                    frames = np.array(frames)
                    await vframe_queue.put((i_batch,frames))
                    i_batch += 1
                    frames = []

        if len(frames) > 0:
            frames = np.array(frames)
            await vframe_queue.put((i_batch,frames))
    
    async def inference_worker(n):
        while True:
            i,frames = await vframe_queue.get()
            frames = pickle.dumps(frames)

            while True:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(CAPTIONS_MODEL_URL, data=frames) as response:
                            captions = await response.text()
                            captions = eval(captions)
                            break
                except (aiohttp.client_exceptions.ClientOSError, aiohttp.client_exceptions.ClientPayloadError) as e:
                    pass

            await caption_queue.put((i,captions))

            if vframe_queue.empty():
                break
                
    async def fetch_captions():
        captions = []
        while True:
            k,caption = await caption_queue.get()
            captions += caption
            if caption_queue.empty():
                return captions
    
    cap = cv.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    info = f'{video_path} with {total_frames} frames ({frame_width}x{frame_height})'
    
    print(info)
    
    n_workers_eff = min(int(np.ceil(total_frames/PREBATCH_SIZE)),N_WORKERS)
    print(f'PREBATCH_SIZE = {PREBATCH_SIZE}; N_WORKERS = {N_WORKERS}; n_workers_eff = {n_workers_eff}')
    
    tasks = []
    tasks += [inference_worker(n) for n in range(n_workers_eff)]
    tasks += [video_reader()]
    _ = await asyncio.gather(*tasks)

    return await fetch_captions()



async def main():
    video_path = '../sample_video/test.mp4'
    
    print(f'captions_model_url={CAPTIONS_MODEL_URL}')
    
    start_time = time.time()
    captions = await caption_video(video_path = video_path)
    end_time = time.time()
    
    print("--- %s seconds ---" % (end_time - start_time))
    print("--- %s frames/sec ---" % (len(captions)/(end_time - start_time)))
    print(f'len(captions)={len(captions)}')
    print('excerpt:')
    for c in captions[::5][:10]:
        print(c.strip('\n'))

asyncio.run(main())