FROM jupyter/minimal-notebook:lab-3.1.17

USER root

COPY ./requirements.txt ${HOME}/requirements.txt

RUN pip install -r requirements.txt --no-cache-dir \
 && jupyter labextension install \
    jupyterlab-system-monitor@0.7.0 \
    jupyterlab-topbar-extension@0.6.0 \
    jupyter-matplotlib@0.10.0 \
 && rm requirements.txt

RUN echo "c.NotebookApp.iopub_data_rate_limit = 10000000" >> /home/jovyan/.jupyter/jupyter_notebook_config.py \
 && echo "c.NotebookApp.iopub_msg_rate_limit = 100000" >> /home/jovyan/.jupyter/jupyter_notebook_config.py \
 && mkdir -m 777 /home/jovyan/.config

USER ${NB_UID}
