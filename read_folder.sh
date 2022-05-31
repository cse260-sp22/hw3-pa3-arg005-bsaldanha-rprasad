#!/bin/bash

while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

#apf.12929116.exp-1-07.nprocs\=64.px\=1.py\=64.i\=25000.n\=2000.k\=0.ref\=0.out

folder_name=${folder_name}
output_file=${output_file:-0}

if [ "$output_file" -eq "0" ]; then
    output_file="$folder_name/output.csv"
fi

echo "n,px,py,k,i,runtime,gflops,Linf,L2" > $output_file

for file in $folder_name/*; do
    echo "$(basename "$file")"
    read n <<< $(awk '/@/{printf "%f", substr($2,1) }' $file)
    read px <<< $(awk '/@/{printf "%f", substr($4,1) }' $file)
    read py <<< $(awk '/@/{printf "%f", substr($5,1) }' $file)
    read k <<< $(awk '/@/{printf "%f", substr($6,1) }' $file)
    read i <<< $(awk '/@/{printf "%f", substr($7,1) }' $file)
    read runtime <<< $(awk '/@/{printf "%f", substr($8,1) }' $file)
    read gflops <<< $(awk '/@/{printf "%f", substr($9,1) }' $file)
    read Linf <<< $(awk '/@/{printf "%f", substr($10,1) }' $file)
    read L2 <<< $(awk '/@/{printf "%f", substr($11,1) }' $file)
    echo "$n,$px,$py,$k,$i,$runtime,$gflops,$Linf,$L2" >> $output_file
done
