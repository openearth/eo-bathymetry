FROM jupyter/minimal-notebook:lab-3.1.17

USER root

# install gcloud
RUN apt-get update -y \
 && apt-get install -y apt-transport-https ca-certificates gnupg curl sudo \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
 && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - \
 && apt-get update -y \
 && apt-get install google-cloud-cli -y

COPY ./environment.yaml ${HOME}/environment.yaml

# create mamba environment
RUN mamba env create -f environment.yaml

# install jupyter labextensions and install geo-env
RUN mamba run -n geo-env jupyter labextension install \
    jupyterlab-system-monitor@0.7.0 \
    jupyterlab-topbar-extension@0.6.0 \
    jupyter-matplotlib@0.10.0 \
 && mamba clean -qafy \
 && rm environment.yaml \
 && mamba run -n geo-env python -m ipykernel install --name=geo-env

RUN echo "c.NotebookApp.iopub_data_rate_limit = 10000000" >> /home/jovyan/.jupyter/jupyter_notebook_config.py \
 && echo "c.NotebookApp.iopub_msg_rate_limit = 100000" >> /home/jovyan/.jupyter/jupyter_notebook_config.py \
 && mkdir -m 777 /home/jovyan/.config

USER ${NB_UID}
