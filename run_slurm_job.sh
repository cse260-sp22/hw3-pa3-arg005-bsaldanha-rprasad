#!/bin/bash
# bash ./run_slurm_job.sh --N 0 --px 1 --py 16 --i 2000 --shared 1 --t 30 --profile 0 --expanse 1
# N is 0 means N0: (n = 800)

while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done


# global variable
N=${N:-0}
px=${px:-1}
py=${py:-16}
i=${i:-10000}
shared=${shared:-0}
expanse=${expanse:-1}
t=${t:-60} # time in seconds
profile=${profile:-0}

if [ "$expanse" -eq "0" ]; then
    target_slurm_file="$(pwd)/sorken.slurm"
else
    target_slurm_file="$(pwd)/expanse.slurm"
fi


get_email() {
    user=$(echo $USER)
    if [ "$expanse" -eq "0" ]; then
        # sorken
        echo "$user@ucsd.edu"
    elif [ $user == "bran451" ]; then
        echo "bsaldanha@ucsd.edu"
    elif [ $user == "raghavprasad" ]; then
        echo "rprasad@ucsd.edu"
    elif [ $user == "fermi" ]; then
        echo "arg005@ucsd.edu"
    else
        echo "arg005@ucsd.edu"
    fi
}

get_nodes() {
    px=$1
    py=$2
    cores_per_node=128
    if [ "$expanse" -eq "0" ]; then
        cores_per_node=32
    fi

    nprocs=$(($px*$py))
    nnodes=$(($nprocs/$cores_per_node))
    if (( $nprocs % $cores_per_node != 0 ))
    then
        nnodes=$(( $nnodes + 1 ))
    fi
    echo $nnodes
}

get_n() {
    N=$1
    n=0
    if [ "$N" -eq "0" ]; then
        n=800
    elif [ "$N" -eq "1" ]; then
        n=2000
    elif [ "$N" -eq "2" ]; then
        n=8000
    fi
    echo $n
}

get_partition_type() {
    if [ "$expanse" -eq "0" ]; then
        # if sorken, return CLUSTER
        echo "CLUSTER"
    elif [ "$shared" -eq "0" ]; then
        echo "compute"
    elif [ "$shared" -eq "1" ]; then
        echo "shared"
    fi
}

pad() {
    echo $1 | awk '{printf "%02d\n", $0;}'
}

convert_seconds() {
    totaltime=$1
    seconds=$((totaltime%60))
    minutes=$((totaltime/60))
    hours=$((minutes/60))
    seconds=$(pad $seconds)
    minutes=$(pad $minutes)
    echo "$hours:$minutes:$seconds"
}

get_n_tasks() {
    px=$1
    py=$2
    nprocs=$(($px*$py))
    nnodes=$(get_nodes $px $py)
    ntasks=$(($nprocs/$nnodes))
    echo $ntasks
}

n=$(get_n $N) # matrix size
nprocs=$(($px*$py))
nodes=$(get_nodes $px $py)
email=$(get_email)
partition_type=$(get_partition_type)
outputfile="apf.%j.%N.nprocs=$nprocs.px=$px.py=$py.i=$i.n=$n.out"
jobtime=$(convert_seconds $t)
n_tasks_per_node=$(get_n_tasks $px $py)

new_command="srun --mpi=pmi2 -n $nprocs .\/apf -n $n -i $i -x $px -y $py"
if [ "$profile" -eq "1" ]; then
    new_command="srun --mpi=pmi2 -n $nprocs tau_exec -io .\/apf -n $n -i $i -x $px -y $py"
fi


echo "Running for nprocs = $nprocs, px = $px, py = $py"

sed -i -e "s/^#SBATCH --partition=.*/#SBATCH --partition=$partition_type/g" $target_slurm_file
sed -i -e "s/^#SBATCH --nodes=.*/#SBATCH --nodes=$nodes/g" $target_slurm_file
sed -i -e "s/#SBATCH --mail-user=.*/#SBATCH --mail-user=$email/g" $target_slurm_file
sed -i -e "s/^#SBATCH --output=.*/#SBATCH --output="$outputfile"/g" $target_slurm_file
sed -i -e "s/^#SBATCH -t.*/#SBATCH -t $jobtime/g" $target_slurm_file
sed -i -e "s/^#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=$n_tasks_per_node/g" $target_slurm_file
sed -i -e "s/^srun --mpi.*$/$new_command/g" $target_slurm_file

if [ "$profile" -eq "0" ]; then
    sed -i -e "s/.*load tau.*//g" $target_slurm_file
else
    echo -n "module load tau" >> $target_slurm_file
fi

sbatch $target_slurm_file
