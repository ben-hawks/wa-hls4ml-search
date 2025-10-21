# Installing on HPRC
```
# $SCRATCH is assumed to be your scratch directory (or wherever you have large amounts of storage) on HPRC
cd $SCRATCH
git clone git@github.com:ben-hawks/wa-hls4ml-search.git
cd wa-hls4ml-search.git
mkdir hlsproj
mkdir hlsproj/output
cd HPRC_scripts
source modules.sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# runner.sh usage
The runner.sh file is used to launch a slurm batch job which runs simulations in parallel.  
You can use the runner.sh script like so:  
```
bash runner.sh <start_config_num> <end_config_num>
```
this will submit a slurm batch job for all configs between [start_config_num, end_config_num). Each config will get 1 core.  
## Example
```
bash runner.sh 1 49
```
This will run configs 1 - 48 (inclusive) in parallel. Each will use 1 core, so in total the script will request 48 cores.  
Note that this is currently hardcoded for 2layer configs only.
