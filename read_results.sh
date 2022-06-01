#!/bin/bash
# bash ./read_folder.sh --folder_name folder_name

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
    read n <<< $(awk '/@/{printf "%d", $2 }' $file)
    read px <<< $(awk '/@/{printf "%d", $4 }' $file)
    read py <<< $(awk '/@/{printf "%d", $5 }' $file)
    read k <<< $(awk '/@/{printf "%s", $6 }' $file)
    read i <<< $(awk '/@/{printf "%d", $7 }' $file)
    read runtime <<< $(awk '/@/{printf "%f", $8 }' $file)
    read gflops <<< $(awk '/@/{printf "%f", $9 }' $file)
    read Linf <<< $(awk '/@/{printf "%f", $10 }' $file)
    read L2 <<< $(awk '/@/{printf "%f", $11 }' $file)
    echo "$n,$px,$py,$k,$i,$runtime,$gflops,$Linf,$L2" >> $output_file
done
