#!/home/yzhou/pyenv/bin/python
from joblib import Parallel, delayed
from os import system
import argparse


parser = argparse.ArgumentParser(description='Parallel run commands from commands file: ./parallel_run.py --file cmd.txt -j 4')
parser.add_argument('-j', '--num_jobs', type=int, default=-1, help='set how many jobs to run in parallel')
parser.add_argument('--file', type=str, required=True, help='commands file')
args = parser.parse_args()

commands = open(args.file).read().splitlines()
Parallel(n_jobs=args.num_jobs)(delayed(system)(cmd) for cmd in commands)
