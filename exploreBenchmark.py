import argparse
import subprocess
from pathlib import Path
import os

CoreFreq = []
CoreVoltage = []
MemoryFreq = []
MemoryVoltage = []

# Parser to collect user given arguments
parser = argparse.ArgumentParser(prog="exploreDVFS", description='Run Benchmark and Perform DVFS Exploration')
group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('-b', '--benchmark', metavar='path', type=str, help="Name of the benchmark", required=True)
parser.add_argument('-c', const=1, default=0, action='store_const', help="Performs exploration on GPU Core")
parser.add_argument('-m', const=1, default=0, action='store_const', help="Performs exploration on GPU Memory")
group.add_argument('-v', const=1, default=0, help="Performs exploration of voltage", action='store_const')
group.add_argument('-f', const=1, default=0, help="Performs exploration of frequency", action='store_const')
parser.add_argument('-lc', '--levelscore', help="Performance Levels to be explored on Core", nargs='+', type=int, choices=range(0, 8))
parser.add_argument('-lm', '--levelsmemory', help="Performance Levels to be explored on Memory", nargs='+', type=int, choices=range(0, 4))
args = parser.parse_args()

# Checks if the benchmark exists
folder = str(args.benchmark[:args.benchmark.rfind('/')])
print(folder)
if not os.path.isdir(folder) or not os.path.isfile(args.benchmark):
	print("Benchmark doesn't exist!")
	exit()

print(Path(folder + "/Results").mkdir(parents=True, exist_ok=True))

bashCommand = "rocm-smi -S"
process = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE, check=True, text=True)
i = 0
for line in process.stdout.split('\n'):
	if i > 4 and i < 13:
		CoreFreq.append(line.replace(':', '').split()[2].replace("Mhz", ''))
		CoreVoltage.append(line.replace(':', '').split()[3].replace("mV", ''))
	if i > 13 and i < 18:
		MemoryFreq.append(line.replace(':', '').split()[2].replace("Mhz", ''))
		MemoryVoltage.append(line.replace(':', '').split()[3].replace("mV", ''))
	i = i + 1

# Exploration of Core and Memory
if args.c == 1 and args.m == 1:
	print("Exploration of Core and Memory")
	if args.v == 1:
		print("	Exploration of volt")
	else:
		print("	Exploration of freq")


# Exploration of Core
elif args.c == 1:
	print("Exploration of Core")
	if args.v == 1:
		print("	Exploration of volt")
	else:
		print("	Exploration of freq")

# Exploration of Memory
elif args.m == 1:
	print("Exploration of Memory")
	if args.v == 1:
		print("	Exploration of volt")
	else:
		print("	Exploration of freq")

else:
	print("No indication of exploration given [v:voltage, f:frequency]")
