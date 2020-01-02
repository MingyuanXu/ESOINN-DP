#!/usr/bin/env python
# coding=utf-8
pbscpustr="""
#!/bin/bash -l
#PBS -l nodes=1:ppn=%d
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -q small
#PBS -N %s 
"""

lsfgpustr="""
#!/bin/sh
#BSUB -q %s
#BSUB -n 1
#BSUB -o %%J.out
#BSUB -e %%J.err
#BSUB -J %s
#BSUB -R "select[ngpus > 0] rusage[ngpus_excl_p=1]"
"""
