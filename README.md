### ML Video captioning tool

A tool for creating video captions in bulk. Uses the [BLIP2 + OPT](https://huggingface.co/Salesforce/blip2-opt-2.7b) img2txt model as the captioning module.

Forms the basic building block in the preprocessing pipeline used in the forthcoming <a>putin-dataset</a> comprising of Vladimir Putin clips. 

#### single-gpu server

Running the single-GPU captioning tool works on a Flask HTTP server.
```
cd flask
pip install -r requirements.txt
./start.sh
```
(loading BLIP2 into memory takes several seconds)

Two example clients are provided: 
* `cli-dummydata.py` (finds captions on a random noise data)
* `cli-putinmask.py` (finds a putin mask in an example video `sample_videos/putin_test.mp4` based on the provided captions)

#### multi-gpu server

Running the captioning tool on multiple GPUs works with torchserve. 
```
cd torchserve
pip install -r requirements-torchserve.txt
./server-create.sh
./start.sh
```
(takes some time to load BLIP2 into GPU memory)

After the server is up and running, several example clients are provided:
* `cli-dummydata.py` (finds captions on a random noise data)
* `cli-single.py` (find captions, uses sync video loading and async single-frame requests)
* `cli-str-single.py` (find captions, uses async stream-like video loading and async single-frame requests)
* `cli-str-prebatch.py` (find captions, uses async stream-like video loading and async prebatched requests)
* `cli-putinmask.py` (finds a putin mask in an example video `sample_videos/putin_test.mp4` based on the provided captions)

##### client benchmark

| client script               | torchserve config file   | capt. speed (frm/s) | video res  | client params |
|:-------------------------------|:-------------------------|:-----------------:|:----------|:--------------|
| cli-single.py     | config-single.properties | 34.6   | 480x270   |               |
| cli-str-single.py    | config-single.properties | 64.5  | 480x270   | |
| **cli-str-prebatch.py** | config.properties        | **140.4**  | 480x270   | prebatch_size = 32 |
| cli-single.py     | config-single.properties | 30.4  | 640x360   | |
| cli-str-single.py    | config-single.properties | 62.9  | 640x360   | |
| **cli-str-prebatch.py** | config.properties        | **125.1**  | 640x360   | prebatch_size = 32 |
| cli-single.py     | config-single.properties | 12.7  | 1280x720  | |
| cli-str-single.py    | config-single.properties | 40.7   | 1280x720  | |
| **cli-str-prebatch.py** | config.properties        | **76.6**  | 1280x720  | prebatch_size = 16 |
| cli-single.py     | config-single.properties | 6.4   | 1920x1080 | |
| cli-str-single.py    | config-single.properties | 29.1  | 1920x1080 | |
| **cli-str-prebatch.py** | config.properties        | **43.2**  | 1920x1080 |  |

Non-standard client parameters in the last column are given. All tests were completed locally i.e. without network latency. A single [rescaled video](https://file-examples.com/index.php/sample-video-files/sample-mp4-files/) with 901 frames was captioned.

##### config.properties

To **change the number of gpus** we must set the `number_of_gpu` parameter and change the `minWorkers`, `maxWorkers` parameters accordingly in the model specification. To use only a subset of gpus available, set the `CUDA_VISIBLE_DEVICES` variable to a list in the `start.sh` script.

### Technology used

* **huggingface transformers** - BLIP2 model
* **flask** - used by the single-gpu server
* **torchserve** - used by the multi-gpu server
* **aiohttp, asyncio** - concurrent runtime in bulk
