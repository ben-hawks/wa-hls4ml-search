# Installing on HPRC
```
cd $SCRATCH
git clone git@github.com:keeganasmith/wa-hls4ml-search.git
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
this will submit a slurm batch job for all configs between [start_config_num, end_config_num). Each config will get 4 cores.  
## Example
```
bash runner.sh 1 49
```
This will run configs 1 - 48 (inclusive) in parallel. Each will use 4 cores, so in total the script will request 4 * 48 = 192 cores.  
Since there are 96 cores / node on ACES, this will request 2 nodes.  
Note that this is currently hardcoded for 2layer configs only.
