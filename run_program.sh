#!/bin/bash
# bash ./run_program.sh --px 1 --py 2 --i 2000 -n 800

while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

px=${px:-2}
py=${py:-2}
n=${n:-100}
i=${i:-100}
nprocs=$(($px*$py))

echo "Running for nprocs = $nprocs, px = $px, py = $py"
# make clean; make
mpirun -n $nprocs ./apf -n $n -i $i -y $py -x $px
