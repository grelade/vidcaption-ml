from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from ts.torch_handler.base_handler import BaseHandler
from abc import ABC
import logging
import pickle
import numpy as np

logger = logging.getLogger(__name__)

class BLIP2Handler(BaseHandler, ABC):
    def __init__(self):
        super(BLIP2Handler, self).__init__()
        self._batch_size = 0
        self.initialized = False
        self.model = None
        
    def initialize(self, ctx):
        self.manifest = ctx.manifest
        self.metrics = ctx.metrics

        logger.info(f"Manifest: {self.manifest}")

        properties = ctx.system_properties
        
        device = "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
    
        self._batch_size = properties["batch_size"]

        logger.info(f"properties: {properties}")

        self.device = torch.device(device)

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", device = self.device)
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.model.to(self.device)
        
        logger.debug("BLIP2 model loaded successfully")

        self.initialized = True
        
    def preprocess(self, frames):
        
        frames_list = []
        for frame in frames:
            frames_list.append(pickle.loads(frame['body']))
        
        logging.info(f"Received data: {len(frames_list)} of type {type(frames_list[0])}")

        frames = np.stack(frames_list)
        inputs = self.processor(frames, return_tensors="pt").to(self.device, torch.float16)
        return inputs

    def inference(self, inputs):
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts   
    
_service = BLIP2Handler()

def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)

        return data
    except Exception as e:
        raise e