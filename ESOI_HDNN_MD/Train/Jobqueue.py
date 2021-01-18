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
pbsgpustr="""
#!/bin/bash -l
#PBS -l nodes=1:ppn=%d:gpus=1
#PBS -l walltime=120:00:00
#PBS -j oe
#PBS -q %s
#PBS -N %s 
"""

lsfcpustr="""
#!/bin/sh
#BSUB -q normal
#BSUB -n %d
#BSUB -o %%J.out
#BSUB -e %%J.err
#BSUB -J %s
#BSUB -R span[hosts=1]
"""

lsfgpustr="""
#!/bin/sh
#BSUB -q %s
#BSUB -n 1
#BSUB -o %%J.out
#BSUB -e %%J.err
#BSUB -J %s
#BSUB -R "rusage[ngpus_physical=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
"""

