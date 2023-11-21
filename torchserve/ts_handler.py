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
        self.inference_mode = None
        
    def initialize(self, context):
        logger.info(f"Manifest: {context.manifest}")
        logger.info(f"properties: {context.system_properties}")
        
        self._batch_size = context.system_properties["batch_size"]
        device = "cuda:" + str(context.system_properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", device = self.device)
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.model.to(self.device)
        
        logger.debug("BLIP2 model loaded successfully")
        self.initialized = True
        
    def preprocess(self, inputs):
        """Very basic preprocessing code - only tokenizes.
        Extend with your own preprocessing steps as needed.
        """
        frames_list = []
        for input_ in inputs:
            input_ = pickle.loads(input_['body'])
            if len(input_.shape)==3:
                self.inference_mode = 'single'
                frames_list.append(input_)
            elif len(input_.shape)==4:
                self.inference_mode = 'batch'
                frames_list.extend(list(input_))
        frames = np.array(frames_list)
        
        logging.info(f"Received data: {len(frames)} of type {type(frames[0])}")
        inputs = self.processor(frames, return_tensors="pt").to(self.device, torch.float16)  

        return inputs

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        if self.inference_mode == 'batch':
            generated_texts = [generated_texts]
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