#!/usr/bin/env bash

# This script is used to build the user image for the Xilinx Docker containers, which the acorn-dac-cicd
# container uses as it's base. The script is intended to be run from the root of the acorn-dac-cicd repository.
# The script will download the Ubuntu ISO and Base tarball images, extract the rootfs
# and create a base image for use in the Xilinx Docker containers.

git submodule update --init --recursive

cd xilinx-docker/recipes/user-images/v2020.1/ubuntu-18.04.2-user

# build the base image
pushd ../../../base-images/ubuntu-18.04.2/
sudo ./fetch_depends.sh --iso
sudo ./build_image.sh --iso
popd

./build_image.sh --iso