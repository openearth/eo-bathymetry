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

USER ${NB_UID}
