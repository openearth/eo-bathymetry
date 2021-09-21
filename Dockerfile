FROM jupyter/minimal-notebook:lab-3.1.11

USER root

COPY ./requirements.txt ${HOME}/requirements.txt

RUN pip install -r requirements.txt --no-cache-dir \
 && jupyter labextension install \
    @jupyterlab/debugger \
    @jupyterlab/toc \
    jupyterlab-system-monitor \
    jupyterlab-topbar-extension \
    jupyter-matplotlib \
 && rm requirements.txt

RUN echo "c.NotebookApp.iopub_data_rate_limit = 10000000" >> /home/jovyan/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.iopub_msg_rate_limit = 100000" >> /home/jovyan/.jupyter/jupyter_notebook_config.py

USER ${NB_UID}
