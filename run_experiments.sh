#!/bin/bash
# two ways to run the script:

# first: give a target time, #iterations would be automatically computed to give `target_time` seconds
# bash ./run_experiments.sh --results_folder results_31may_evening --target_time 10 --input_file q2c.csv

# second way: give a fixed #iterations
# bash ./run_experiments.sh --results_folder results_31may_evening --iterations 100000

while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

results_folder=${results_folder:-0}
compute=${compute:-1}
ref=${ref:-0}
input_file=${input_file:-0}

# either give target_time or iterations
# if target_time is give, #iterations would be computed assuming 100% scaling (reference: 20gflops for 1 core)
# if nothing is provided $target_time would be set to 10s
target_time=${target_time:-10}
iterations=${iterations:-0}



user=$(echo $USER)
if [ "$results_folder" == "0" ]; then
    results_folder=$(echo results_$(date +"%Y-%m-%dT%H:%M%z")_$user)
fi

if [ "$input_file" -eq "0" ]; then
	input_file="experiment_input.csv"
fi

while IFS=',' read -ra array; do
  pxarray+=("${array[0]}")
  pyarray+=("${array[1]}")
  Narray+=("${array[2]}")
  karray+=("${array[3]}")
#   iarray+=("${array[4]}")
done < $input_file

# printf '%s\n' "${pxarray[@]}"

get_iterations() {
    #gflops = 28*n^2*niters*1e9/time
    #niters = time*gflops*1e9/(28*n^2)
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

	# 20gflops is estimated perf for 1 core!
	single_core_gflops=20
    estimated_gflops=$(($single_core_gflops*$px*$py))
    niters=$(($estimated_gflops*$target_time*1000/$n*1000/$n*1000/28))
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

    echo "Running for px = $px, py = $py, N = $N, k = $k, i = $iters, compute = $compute,  ref = $ref"

    bash ./run_slurm_job.sh --N $N --px $px --py $py --i $iters --compute 1 --t 30 --k $k --results_folder $results_folder --compute $compute --ref $ref
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

echo "Run the following to collect results in $results_folder/output.csv file. once the job is completed"
echo "bash ./read_results.sh --folder_name $results_folder"

