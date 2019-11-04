#!/usr/bin/env python
# coding=utf-8
pbsstr="""
#!/bin/bash -l
#PBS -l nodes=1:ppn=%d
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -q small
#PBS -N %s 
"""
