# Get started with AIchor

This is an AIchor demo project, please fell free for fork it if you intend on trying it out.

## Goal

This project aims to get up to speed with AIchor by going through the whole process.


## How to use it ?

You can find multiple manifests samples in the `aichor_manifests` directories. If you want to try hugging face accelerate for example, all you need to do is to copy it:

```bash

$ git add aichor_manifests/hugging-face-accelerate/rdma/1_worker/manifest.1wrkr-1-h100-80gb.yaml
$ git commit -m "aichor[hugging-face-accelerate/rdma/1_worker/manifest.1wrkr-1-h100-80gb.yaml]: experiment" # commit has "aichor[<manifest-path>]: " to trigger experiment where `<manifest-path>` is the path of manifest relative to the `aichor_manifests/` directory

$ git push
```

# Demo projects

## Smoke test

This project works accross all AIchor operators. It runs a vanilla experiment:
- print chosen operator environment variables
- creates a tensorboard log with the commit message
- sleeps for x seconds

## Hugging face Accelerate

Use hugging face accelerate to setup the distribution with pytorch operator.

## Jax demo

Demo project using jax distributed with processes spread accross multiple containers.

## Parallel jobs demos

Run multiple jobs in parallel in a single AIchor experiment. Each job being a container. Using TF operator.

## PyTorch demo

Demo project using pytorch distributed with processes spread accross multiple containers.

## raytune demo

Demo project using ray[tune], distributed accross multiple containers thanks to kuberay.

## xgboost demo

Demo project using xgboost distributed with processes spread accross multiple containers.