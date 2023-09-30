# LLM API

This application can be used to run LLMs (Large Language Models) in docker containers, it's built in a generic way so that it can be reused for multiple types of models.


# Usage

In order to run this API on a local machine, a running docker engine is needed.

run using docker:

create a `config.yaml` file with the configs described below and then run:

```
docker run -v $PWD/models/:/models:rw -v $PWD/config.yaml:/llm-api/config.yaml:ro -p 8000:8000 --ulimit memlock=16000000000 1b5d/llm-api
```

or use the `docker-compose.yaml` in this repo and run using compose:

```
docker compose up
```

When running for the first time, the app will download the model from huggingface based on the configurations in `setup_params` and name the local model file accordingly, on later runs it looks up the same local file and loads it into memory

## Config

to configure the application, edit `config.yaml` which is mounted into the docker container, the config file looks like this:

```
models_dir: /models     # dir inside the container
model_family: llama
setup_params:
  key: value
model_params:
  key: value
```

`setup_params` and `model_params` are model specific, see below for model specific configs.

You can override any of the above mentioned configs using environment vars prefixed with `LLM_API_` for example: `LLM_API_MODELS_DIR=/models`

## Endpoints

In general all LLMs will have a generalized set of endpoints

```
POST /generate
{
    "prompt": "What is the capital of France?",
    "params": {
        ...
    }
}
```
```
POST /agenerate
{
    "prompt": "What is the capital of France?",
    "params": {
        ...
    }
}
```
```
POST /embeddings
{
    "text": "What is the capital of France?"
}
```


## Llama / Alpaca on GPU - using GPTQ-for-LLaMa

**Note**: According to [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), you might want to install the [NVIDIA Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) on your host machine. Verify that your nvidia environment is properly by running this:

```
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi
```

You should see a table showing you the current nvidia driver version and some other info:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 11.7     |
|-----------------------------------------+----------------------+----------------------+
...
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

You can run the Llama model using GPTQ-for-LLaMa 4 bit quantization, you can use a docker image specially built for that purpose `1b5d/llm-api:x.x.x-gpu` instead of the default image.

a separate docker-compose file is also available to run this mode:

```
docker compose -f docker-compose.gpu.yaml up
```

or by directly running the container:

```
docker run --gpus all -v $PWD/models/:/models:rw -v $PWD/config.yaml:/llm-api/config.yaml:ro -p 8000:8000 1b5d/llm-api:x.x.x-gpu
```

**Note**: `llm-api:x.x.x-gptq-llama-cuda` and `llm-api:x.x.x-gptq-llama-triton` images have been deprecated, please switch to the `1b5d/llm-api:x.x.x-gpu` image when gpu support is required

Example config file:

```
models_dir: /models
model_family: gptq_llama
setup_params:
  repo_id: user/repo_id
  filename: <model.safetensors or model.pt>
model_params:
  group_size: 128
  wbits: 4
  cuda_visible_devices: "0"
  device: "cuda:0"
```

Example request:

```
POST /generate

curl --location 'localhost:8000/generate' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "What is the capital of paris",
    "params": {
        "temp": 0.8,
        "top_p": 0.95,
        "min_length": 10,
        "max_length": 50
    }
}'
```




