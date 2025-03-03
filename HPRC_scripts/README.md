# Installing on HPRC
```
cd $SCRATCH
git clone git@github.com:keeganasmith/wa-hls4ml-search.git
cd wa-hls4ml-search.git
cd HPRC_scripts
source modules.sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# 2layer_slurm.sh usage
You can use the 2layer_slurm script like so:
```
sbatch 2layer_slurm.sh <start_config_num> <end_config_num>
```
This will spawn processes for each config id, with each process using 4 cores. Example:
```
sbatch 2layer_slurm.sh 1 49
```
This will run configs 1 - 48 (inclusive) in parallel. Each will use 4 cores, so in total the script will request 4 * 48 = 192 cores.  
Since there are 96 cores / node on ACES, this will request 2 nodes.
