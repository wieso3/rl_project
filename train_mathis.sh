#!/bin/bash

#SBATCH --mail-user=kruse@tnt.uni-hannover.de
#SBATCH --mail-type=NONE                # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=dgn_train           # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=slurmlogs/slurm-%j-out.txt      # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)
#SBATCH --time=0-23                     # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS oder Tage-Stunden)
#SBATCH --partition=cpu_normal_stud  # Partition auf der gerechnet werden soll (Bei GPU Jobs unbedingt notwendig)
#SBATCH --cpus-per-task=4              # Reservierung von 4 Threads
#SBATCH --mem=10G                       # Reservierung von 10 GB RAM Speicher pro Knoten
                   # Reservierung von einer GPU. Es kann ein bestimmter Typ angefordert werden:
                                       #SBATCH --gres=gpu:pascal:1

echo "Script starts..."
working_dir=/home/kruse/Schreibtisch/rl/rl_project
echo $working_dir
cd $working_dir

srun ~/anaconda3/tmp/envs/rl/bin/python battle.py -env=battlefield
