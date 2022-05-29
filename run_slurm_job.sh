#!/bin/bash
# bash ./run_slurm_job.sh --N 0 --px 1 --py 16 --i 2000 --shared 1
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

if [ "$expanse" -eq "0" ]; then
    target_slurm_file="$(pwd)/sorken.slurm"
else
    target_slurm_file="$(pwd)/expanse.slurm"
fi

get email() {
    user=$(echo $USER)
    if [ "$expanse" -eq "0" ]; then
        # sorken
        echo "$(user)@ucsd.edu"
    elif [ $user -eq "bran451" ]; then
        echo "bsaldanha@ucsd.edu"
    elif [ $user -eq "raghavprasad" ]; then
        echo "rprasad@ucsd.edu"
    elif [ $user -eq "fermi" ]; then
        echo "arg005@ucsd.edu"
    else
        echo "arg005@ucsd.edu"
    fi
}

get_nodes() {
    px=$1
    py=$2
    nprocs=$((px*py))
    nnodes=$((nprocs/128))
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
    if [ "$shared" -eq "0" ]; then
        echo "compute"
    elif [ "$shared" -eq "1" ]; then
        echo "shared"
    fi
}

n=$(get_n $N)
nprocs=$(($px*$py))
nodes=$(get_nodes $px $py)
email=$(get_email)
partition_type=$(get_partition_type)
new_command="srun --mpi=pmi2 -n $nprocs ./apf -n $n -i $n -x $px -y $py"

echo "Running for nprocs = $nprocs, px = $px, py = $py"

sed -i -e "s/^#SBATCH --partition=.*/SBATCH --partition=$partition_type/g" $target_slurm_file
sed -i -e "s/^#SBATCH --nodes=.*/SBATCH --nodes=$nodes/g" $target_slurm_file
sed -i -e "s/#SBATCH --mail-user=.*/SBATCH --mail-user=$email/g" $target_slurm_file
sed -i -e "s/^srun.*/$newcommand$" $target_slurm_file

sbatch $target_slurm_file
