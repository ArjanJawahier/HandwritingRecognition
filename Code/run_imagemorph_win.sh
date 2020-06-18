#!/bin/bash
root_dir="../Train_Data"
char_dirs="../Train_Data/*"
for char_dir in $char_dirs
    do 
        char_files=($char_dir/*)
        n_files=$(ls $char_dir -1 | wc -l)
        let n_needed=300-$n_files
        for i in $(seq 1 $n_needed)
            do 
                index=$((i%n_files))
                ./imagemorph 1 3 < ${char_files[$index]} > $char_dir/morphed$i.pgm;
        done
done