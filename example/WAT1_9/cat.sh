#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9
do 
    for j in 0 1 2 3 4 5 7
    do 
        cat WAT${i}_300/WAT${i}_300_${j}.mdout |awk '{print $16}' > WAT${i}_300/Moddev_${j}.dat
    done
done 
