singularity exec -B "$PWD":/workspace \
  /opt/singularity/datascience-notebook-deepagents.sif \
  bash -c "cd /workspace && bash"
