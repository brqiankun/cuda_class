#! /bin/bash
current_dir=$(pwd)
echo "there are $(ls | wc -l) items here"
echo "profile $1"
echo $current_dir
# echo $current_dir/report_$(date '+%m%d%H%M').qderp

nsys profile --backtrace=fp --cudabacktrace=all --cuda-memory-usage=true --opengl-gpu-workload=true --force-overwrite=true --output=$current_dir/report_$(date '+%m%d%H%M') $1 
