1) Download ICPSR dataset from https://www.icpsr.umich.edu/web/ICPSR/studies/22140 and place the tsv files into `data` folder

2) Run the following commands on a SLURM server:

uv venv --python=3.12

source .venv/bin/activate

uv pip install -r requirements.txt

sbatch run.slurm

python3 plot_figures.py

