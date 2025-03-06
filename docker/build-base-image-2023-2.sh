#!/usr/bin/env bash

# This script is used to build the user image for the Xilinx Docker containers, which the acorn-dac-cicd
# container uses as it's base. The script is intended to be run from the root of the acorn-dac-cicd repository.
# The script will download the Ubuntu ISO and Base tarball images, extract the rootfs
# and create a base image for use in the Xilinx Docker containers.

git submodule update --init --recursive

ls

cd ./xilinx-docker/recipes/user-images/v2023.2/ubuntu-20.04.4-user

# build the base image
pushd ../../../base-images/ubuntu-20.04.4/
sudo ./fetch_depends.sh --iso
sudo ./build_image.sh --iso
popd

./build_image.sh --iso

docker builder prune -f

rm -rf ./xilinx-docker/recipes/base-images/ubuntu-20.04.4/depends