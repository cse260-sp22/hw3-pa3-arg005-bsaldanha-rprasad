#!/bin/bash
# bash ./run_experiments.sh --results_folder results_31may_evening --target_time 10

while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

results_folder=${results_folder:-0}
# either give target_time or iterations
target_time=${target_time:-10}
iterations=${iterations:-0}

user=$(echo $USER)
if [ "$results_folder" == "0" ]; then
    results_folder=$(echo results_$(date +"%Y-%m-%dT%H:%M%z")_$user)
fi

while IFS=',' read -ra array; do
  pxarray+=("${array[0]}")
  pyarray+=("${array[1]}")
  Narray+=("${array[2]}")
  karray+=("${array[3]}")
#   iarray+=("${array[4]}")
done < experiment_input.csv                                                       

# printf '%s\n' "${pxarray[@]}"

get_iterations() {
    #gflops = 28*n^2*niters*1e9/time
    #time = 28*n^2*niters/gflops/1e9
    #niters = time*gflops*1e9/(28*n^2)
    #if time = 5s
    N=$1
    px=$2
    py=$3

    n=800
    if [ "$N" -eq "0" ]; then
        n=800
    elif [ "$N" -eq "1" ]; then
        n=2000
    else
        n=8000
    fi

    estimated_gflops=$((20*$px*$py))
    niters=$(($estimated_gflops*$target_time*1000/$n*1000/$n*1000/28/1000*1000))
    echo $niters
    # echo "Estimated gflops: $estimated_gflops"
}

run_slurm_job() {
    px=$1
    py=$2
    N=$3
    k=$4

    # computing #iterations to run
    iters=$iterations
    if [ "$iters" -eq "0" ]; then
        iters=$(get_iterations $N $px $py)
    fi

    echo "Running for px = $px, py = $py, N = $N, k = $k, i = $iters"

    ./run_slurm_job.sh --N $N --px $px --py $py --i $i --compute 1 --t 30 --k $k --results_folder $results_folder
}

echo "length of pxarray = ${#pxarray[@]}"
len=${#pxarray[@]}

# iterate i from 1 to length of pxarray
for (( i=1; i<$len; i++ )); do
    px=${pxarray[$i]}
    py=${pyarray[$i]}
    N=${Narray[$i]}
    k=${karray[$i]}
    # ./run_slurm_job.sh --px $px --py $py --i $i --shared 1 --t 60
    run_slurm_job $px $py $N $k
done

