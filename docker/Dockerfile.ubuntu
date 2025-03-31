FROM xilinx-ubuntu-18.04.2-user:v2020.1 AS build
LABEL authors="bhawks"

# Install Miniforge
USER root
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/conda \
    && source "/opt/conda/etc/profile.d/conda.sh" \
    && echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh \
USER xilinx
ENV PATH=/opt/conda/bin:$PATH
#setups stuff
RUN echo "source activate base" > ~/.bashrc
#install packages
COPY files/hls4ml.yml /tmp/hls4ml.yml
RUN conda env create --name hls4ml --file /tmp/hls4ml.yml

RUN conda install -y -c conda-forge conda-pack

RUN conda-pack -n hls4ml -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

RUN /venv/bin/conda-unpack

FROM xilinx-ubuntu-18.04.2-user:v2020.1 AS deploy

COPY --from=build --chown=xilinx /venv /venv

USER xilinx
WORKDIR /home/xilinx

RUN echo "source /venv/bin/activate" >> ~/.bashrc

RUN cat ~/.bashrc

# Add Tini - https://stackoverflow.com/questions/55733058/vivado-synthesis-hangs-in-docker-container-spawned-by-jenkins
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN sudo chmod +x /tini

SHELL ["/bin/bash", "-c", "-l"]
ENTRYPOINT ["/tini", "--"]
CMD ["/bin/bash"]