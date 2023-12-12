FROM quay.io/jupyter/minimal-notebook:lab-4.0.9

USER root

# install gcloud
RUN apt-get update -y \
 && apt-get install -y apt-transport-https ca-certificates gnupg curl sudo \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
 && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - \
 && apt-get update -y \
 && apt-get install google-cloud-cli -y

# create a new mamba environment following:
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#add-a-custom-conda-environment-and-jupyter-kernel
COPY --chown=${NB_UID}:${NB_GID} environment.yaml /tmp/

# create mamba environment
RUN mamba env create -p "${CONDA_DIR}/envs/${env_name}" -f /tmp/environment.yaml && \
    mamba clean --all -f -y

RUN "${CONDA_DIR}/envs/${env_name}/bin/python" -m ipykernel install --user --name="${env_name}" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN \
    # This changes a startup hook, which will activate the custom environment for the process
    echo conda activate "${env_name}" >> /usr/local/bin/before-notebook.d/10activate-conda-env.sh && \
    # This makes the custom environment default in Jupyter Terminals for all users which might be created later
    echo conda activate "${env_name}" >> /etc/skel/.bashrc && \
    # This makes the custom environment default in Jupyter Terminals for already existing NB_USER
    echo conda activate "${env_name}" >> "/home/${NB_USER}/.bashrc"

USER ${NB_UID}
