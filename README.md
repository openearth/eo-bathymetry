# Run notebooks locally

Build the jupyter notebook docker image:
```docker build -t eo-bathymetry .```
Then run the notebook and attach this repository as a volume. Run expose the notebook server on port 8888 on your local machine:
```docker run -p 8888:8888 -v $(pwd):/home/jovyan/work/ eo-bathymetry ```
If you want to run with local gcloud credentials, run with:
```docker run -p 8888:8888 -v $(pwd):/home/jovyan/work/ -v ~/.config/gcloud:/home/jovyan/.config/gcloud eo-bathymetry```
