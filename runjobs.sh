#!/bin/bash

#30,60,100,20 #reminder to do the other system sizes later
for i in {100..100}
do
    for j in {1..4}
    do
        for k in {1..5}
        do
            for l in {1..1}
            do
                cd "L=$i/$j/$k/$l";
                pwd;
                cp ../../../../kmcres.py ./;
                cp ../../../../runkmc_restart.py ./;
                sbatch saviosubmitjob.sh
                #parallel -j 32 < newscript_list.txt
                cd ../../../../;
            done
        done
    done
done
