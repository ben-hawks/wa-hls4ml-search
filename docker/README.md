## wa-hls4ml Docker files

This directory contains Dockerfiles and related scripts to build Docker images for running wa-hls4ml.

### Available Docker Images
The following Docker images are available for use from the ghcr contianer repo. See the packages page of this repo for details on how to pull the images
- **wa-hls4ml-search:v2023.2**: This image is based on Vivado 2023.2 and includes all necessary dependencies to run wa-hls4ml. It is the recommended image for new experiments.
- **wa-hls4ml-search:v2020.1**: This image is based on Vivado 2020.1 and is included for historical purposes. It is not recommended for new experiments and is considered deprecated.

### Building the Docker Images
To build the Docker images locally, navigate to this directory and run the following commands:
```bash
# Build the Vivado 2023.2 based image
source build_base_image-2023-2.sh
docker build -t wa-hls4ml-search:v2023.2 -f Dockerfile.ubuntu-2023.2 . 
```

```bash
# Build the Vivado 2020.1 based image
source build_base_image-2020.sh
docker build -t wa-hls4ml-search:v2020.1 -f Dockerfile.ubuntu .
```

### Using the Docker Images

#### Important Note
Because of some issues with vivado, there are a few important considerations when using these docker images:
- Always run the containers with the --init flag to ensure proper signal handling.
- always set the LD_PRELOAD environment variable to `/lib/x86_64-linux-gnu/libudev.so.1` inside the container to avoid segfault issues inside the container when running vivado.
- set the `XILINXD_LICENSE_FILE` and `LM_LICENSE_FILE` to point to the license server you are using for Vitis/Vivado. Make sure that these servers are accessible from within the container, otherwise adjust the network settings of the container accordingly.

#### Running the Docker Containers
Once the Docker images are built or pulled from the container registry, you can find examples of running them in the [kube](../kube/) directory. These job files demonstrate how to run wa-hls4ml within a k8s cluster, including mounting necessary volumes and setting environment variables.

