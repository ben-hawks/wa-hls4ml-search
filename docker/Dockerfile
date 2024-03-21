FROM bhawks/vivado-docker:alma8-vivado-2020.1-minimal as build
LABEL authors="bhawks"

# Install Miniforge
USER root
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/conda \
    && source "/opt/conda/etc/profile.d/conda.sh" \
    && echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh \
USER vivado
ENV PATH /opt/conda/bin:$PATH
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

FROM bhawks/vivado-docker:alma8-vivado-2020.1-minimal as deploy

COPY --from=build /venv /venv

USER vivado
WORKDIR /home/vivado

RUN echo "source /venv/bin/activate" > ~/.bashrc

RUN cat ~/.bashrc

ENTRYPOINT python -c 'help("modules")'