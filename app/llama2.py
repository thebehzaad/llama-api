"""

Llama 2 implementation on GPU
Created by: BA
Date: Oct 1st 23

params = settings.model_params


"""

#%% Importing libs and utils

import logging
import os
import sys
from typing import AsyncIterator, Dict, List

import huggingface_hub
import torch  # pylint: disable=import-error
from safetensors.torch import load_file as safe_load
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import no_init_weights

from base import BaseLLM
from config import settings

from converting_to_hf.convert_llama_weights_to_hf import convert


#%% 

sys.path.append(os.path.join(os.path.dirname(__file__), "GPTQ-for-LLaMa"))

logger = logging.getLogger("llm-api.gptq_llama")


class LlamaLLM(BaseLLM):

    def __init__(self, params: Dict[str, str]) -> None:
        
        model_path = self._setup()
        group_size = params.get("group_size", 128)
        wbits = params.get("wbits", 4)
        cuda_visible_devices = params.get("cuda_visible_devices", "0")
        dev = params.get("device", "cuda:0")

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.device = torch.device(dev)

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.setup_params["repo_id"], use_fast=False)

    def _setup(self):
        
        model_dir = super().get_model_dir(
            settings.models_dir,
            settings.model_family,
            settings.setup_params["filename"],
        )
        model_path = os.path.join(
            model_dir,
            settings.setup_params["filename"],
        )

        #self._download(model_path, model_dir)

        logger.info("setup done successfully for %s", model_path)
        
        return model_path


    def _convert_to_hf(self, input_dir, model_size, output_dir, safe_serialization_flag):
        
        convert(input_dir, model_size, output_dir, safe_serialization_flag)

    def _quantize(self):

        raise NotImplementedError("agenerate endpoint is not yet implemented")

    def _download(self, model_path, model_dir):  # pylint: disable=duplicate-code
        
        if os.path.exists(model_path):
            logger.info("found an existing model %s", model_path)
            return

        logger.info("downloading model to %s", model_path)
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        huggingface_hub.hf_hub_download(
            repo_id=settings.setup_params["repo_id"],
            filename=settings.setup_params["filename"],
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(models_dir, ".cache"),
        )

        huggingface_hub.hf_hub_download(
            repo_id=settings.setup_params["repo_id"],
            filename="config.json",
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(models_dir, ".cache"),
        )

        huggingface_hub.hf_hub_download(
            repo_id=settings.setup_params["repo_id"],
            filename="tokenizer.model",
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(models_dir, ".cache"),
        )

        huggingface_hub.hf_hub_download(
            repo_id=settings.setup_params["repo_id"],
            filename="tokenizer_config.json",
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(models_dir, ".cache"),
        )

    def generate(self, prompt: str, params: Dict[str, str]) -> str:
        
        """
        Generate text from Llama using the input prompt and parameters
        """

        min_length = params.get("min_length", 10)
        max_length = params.get("max_length", 50)
        top_p = params.get("top_p", 0.95)
        temperature = params.get("temp", 0.8)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                min_length=min_length,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
            )
        return self.tokenizer.decode(
            [el.item() for el in generated_ids[:, input_ids.shape[1] :][0]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    async def agenerate(
        self, prompt: str, params: Dict[str, str]
    ) -> AsyncIterator[str]:
        """
        Generate text stream from Llama using the input prompt and parameters
        """
        raise NotImplementedError("agenerate endpoint is not yet implemented")
        if False:  # pylint: disable=using-constant-test,disable=unreachable
            yield

    def embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for Llama using the input text
        """
        raise NotImplementedError("embeddings endpoint is not yet implemented")
