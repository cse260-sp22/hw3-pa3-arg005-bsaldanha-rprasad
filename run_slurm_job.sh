#!/bin/bash
# bash ./run_slurm_job.sh --N 0 --px 1 --py 16 --t 30
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
i=${i:-8000}
# expanse=${expanse:-1}
t=${t:-60} # time in seconds
profile=${profile:-0}
k=${k:-0}
results_folder=${results_folder:-0}
ref=${ref:-0}
debug=${debug:-0}
compute=${compute:-0} # run on compute node

expanse=1

is_sorken=$(hostname | grep sorken | wc -c)
if [ $is_sorken -gt 0 ]; then
	expanse=0
fi

echo "expanse = $expanse"

if [ "$expanse" -eq "0" ]; then
    target_slurm_file="$(pwd)/sorken.slurm"
else
    target_slurm_file="$(pwd)/expanse.slurm"
fi

binaryfile=".\/apf"
if [ "$ref" -eq "1" ]; then
    binaryfile="\/share\/public\/cse260-sp22\/HW\/hw3\/apf-ref"
fi

user=$(echo $USER)
if [ "$results_folder" == "0" ]; then
    results_folder=$(echo results_$(date +"%Y-%m-%dT%H:%M%z")_$user)
fi

mkdir $results_folder

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
    nprocs=$1
    if [ "$expanse" -eq "0" ]; then
        # if sorken, return CLUSTER
        echo "CLUSTER"
    elif [ "$compute" -eq "1" ]; then
        echo "compute"
    elif [ "$nprocs" -ge "128" ]; then
        echo "compute"
    else
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
outputfile="$results_folder\/apf.%j.%N.nprocs=$nprocs.px=$px.py=$py.i=$i.n=$n.k=$k.ref=$ref.out"
jobtime=$(convert_seconds $t)
n_tasks_per_node=$(get_n_tasks $px $py)
partition_type=$(get_partition_type $nprocs)

new_command="srun --mpi=pmi2 -n $nprocs $binaryfile -n $n -i $i -x $px -y $py"
if [ "$expanse" -eq "0" ]; then
    new_command="mpirun -n $nprocs $binaryfile -n $n -i $i -x $px -y $py"
fi

# if [ "$profile" -eq "1" ]; then
#     new_command="srun --mpi=pmi2 -n $nprocs tau_exec -io .\/apf -n $n -i $i -x $px -y $py"
# fi

if [ "$k" -eq "1" ]; then
    new_command="$new_command -k"
fi

if [ "$debug" -eq "1" ]; then
    new_command="$new_command -d"
fi

echo "Running for nprocs = $nprocs, px = $px, py = $py"

sed -i -e "s/^#SBATCH --partition=.*/#SBATCH --partition=$partition_type/g" $target_slurm_file
sed -i -e "s/^#SBATCH --nodes=.*/#SBATCH --nodes=$nodes/g" $target_slurm_file
sed -i -e "s/#SBATCH --mail-user=.*/#SBATCH --mail-user=$email/g" $target_slurm_file
sed -i -e "s/^#SBATCH --output=.*/#SBATCH --output="$outputfile"/g" $target_slurm_file
sed -i -e "s/^#SBATCH -t.*/#SBATCH -t $jobtime/g" $target_slurm_file
sed -i -e "s/^#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=$n_tasks_per_node/g" $target_slurm_file

if [ "$expanse" -eq "0" ]; then
	sed -i -e "s/^mpirun.*$/$new_command/g" $target_slurm_file
else
	sed -i -e "s/^srun.*$/$new_command/g" $target_slurm_file
fi

if [ "$profile" -eq "0" ]; then
    sed -i -e "s/.*load tau.*//g" $target_slurm_file
else
    echo -n "module load tau" >> $target_slurm_file
fi

sbatch $target_slurm_file
