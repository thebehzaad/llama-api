# LLAMA2 API

This application can be used to run Llama2 in docker containers.


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



