# steps to run models on V2

1. Either store the model on an object storage, for [local mode](https://docs.seldon.io/projects/seldon-core/en/v2.0.0/contents/getting-started/docker-installation/index.html#local-models) do:
```bash
mkdir ~/storage
export LOCAL_MODEL_FOLDER=~/storage
make deploy-local
```

2. Copy the content into the object storage or local storage, for local storage
```
cp -r audio ~/storage
```

3. 
