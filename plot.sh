mkdir -p /workspace/.cache/matplotlib
export MPLCONFIGDIR=/workspace/.cache/matplotlib

python -m src.run_direct \
  --ref /workspace/inputs/ref.png \
  --data /workspace/inputs/data.csv \
  --iters 6 \
  --page 0 \
  --data_type generic \
  --plot_types scatterplot,boxplot
