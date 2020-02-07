import argparse
import subprocess
from pathlib import Path
import os

CoreFreq = []
CoreVoltage = []
MemoryFreq = []
MemoryVoltage = []

def benchmarkCommand(benchmark, folder, levelCore, levelMem, typeEx, explore):
	global CoreFreq
	global CoreVoltage
	global MemoryFreq
	global MemoryVoltage
	name = str(benchmark[0]).replace(str(folder) + '/', '')
	name = name[:name.rfind('.')]
	benchmark = ' '.join(benchmark)

	return str(benchmark) + " 2>&1 | tee " + str(folder) + "/Results/" + name + "-" + str(typeEx) + "-" + str(explore) + "-" + "Core-" + str(levelCore) + "-" + str(CoreFreq[levelCore]) + '-' + str(CoreVoltage[levelCore]) + "-Memory-" + str(levelMem) + '-' + str(MemoryFreq[levelMem]) + '-' + str(MemoryVoltage[levelMem]) + ".txt"

def runBashCommand(bashCommand):
	"""Runs a bash command

	Args:
		bashCommand: Command to be run

	Returns:
		The resulting process
	"""
	process = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE, check=True, text=True)
	return process

def runDVFSscript(command):
	"""Runs the DVFS script

	Args:
		command: Command to be run

	Returns:
		The resulting process
	"""
	script = "./DVFS " + str(command)
	process = subprocess.run(script.split(), stdout=subprocess.PIPE, check=True, text=True)
	return process

def setPerformanceLevel(source, level):
	"""Sets a given performance level for the GPU Core and Memory.

	Args:
		source: string containing word "core" or "mem"
		level: an integer between 0-7 for core and 0-3 memory
	"""
	if source == "core":
		assert level in list(range(0, 8)),"Core Performance Level betwen 0 and 7."
		result = runDVFSscript("-P " + str(level))
		if "ERROR" in result.stdout:
			return False
	elif source == "mem":
		assert level in list(range(0, 4)),"Core Performance Level betwen 0 and 3."
		result = runDVFSscript("-p " + str(level))
		if "ERROR" in result.stdout:
			return False
	else:
		print("Not valid source used - core or mem")

def editPerformanceLevel(source, level, frequency, voltage):
	"""Edits a given performance level for the GPU Core and Memory.

	Args:
		source: string containing word "core" or "mem"
		level: an integer between 0-7 for core and 0-3 memory
		frequency: an integer indicating the frequency
		voltage: an integer indicating the voltage
	"""
	if source == "core":
		assert level in list(range(0, 8)),"Core Performance Level betwen 0 and 7."
		result = runDVFSscript("-L " + str(level) + " -F " +  str(frequency) + " -V " + str(voltage))
		if "ERROR" in result.stdout:
			return False
	elif source == "mem":
		assert level in list(range(0, 4)),"Core Performance Level betwen 0 and 3."
		result = runDVFSscript("-l " + str(level) + " -f " +  str(frequency) + " -v " + str(voltage))
		if "ERROR" in result.stdout:
			return False
	else:
		print("Not valid source used - core or mem")

	return True

def currentPerfLevel():
	global CoreFreq
	global MemoryFreq
	result = runBashCommand("rocm-smi")
	core = -1
	mem = -1
	line = result.stdout.split('\n')
	line = line[5].split(" ")
	core = line[10].replace("Mhz", '')
	mem = line[12].replace("Mhz", '')
	return CoreFreq.index(core), MemoryFreq.index(mem)

def appendCurrentTemp(file):
	result = runBashCommand("rocm-smi")
	line = result.stdout.split('\n')
	temp = line[5].split(" ")[4]

	file = file.split("tee ", 1)[1]
	print(file)
	exit()

	with open(file, "a") as benchFile:
		benchFile.write("appended text")

	return temp

# Parser to collect user given arguments
parser = argparse.ArgumentParser(prog="exploreDVFS", description='Run Benchmark and Perform DVFS Exploration')
group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('-b', '--benchmark', metavar='path', type=str, help="Name of the benchmark", required=True, nargs='+')
parser.add_argument('-c', const=1, default=0, action='store_const', help="Performs exploration on GPU Core")
parser.add_argument('-m', const=1, default=0, action='store_const', help="Performs exploration on GPU Memory")
group.add_argument('-v', const=1, default=0, help="Performs exploration of voltage", action='store_const')
group.add_argument('-f', const=1, default=0, help="Performs exploration of frequency", action='store_const')
parser.add_argument('-lc', '--levelscore', help="Performance Levels to be explored on Core", nargs='+', type=int, choices=range(0, 8))
parser.add_argument('-lm', '--levelsmemory', help="Performance Levels to be explored on Memory", nargs='+', type=int, choices=range(0, 4))
args = parser.parse_args()
if args.levelscore == None:
	args.levelscore = list(range(0,8))

if args.levelsmemory == None:
	args.levelsmemory = list(range(0,4))

# Checks if the benchmark exists and create a Results folder
folder = str(args.benchmark[0][:args.benchmark[0].rfind('/')])
if not os.path.isdir(folder) or not os.path.isfile(args.benchmark[0]):
	print("Benchmark doesn't exist!")
	exit()
Path(folder + "/Results").mkdir(parents=True, exist_ok=True)

# Reset GPU Power Table
result = runBashCommand("rocm-smi -r")
if "Successfully" not in result.stdout:
	print("Not able to reset GPU")
	exit()

# Set GPU fan to 100%
result = runBashCommand("rocm-smi --setfan 255")
if "Successfully" not in result.stdout:
	print("Not able to set fan")
	exit()

# Disable DVFS 
result = runBashCommand("rocm-smi --setperflevel manual")
if "Successfully" not in result.stdout:
	print("Not able to set manual performance level")
	exit()


# Get the current power table
process = runBashCommand("rocm-smi -S")
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
	# Run the benchmark for the proposed levels
	for levels in args.levelscore:
		working = [1] * 8
		while 1 in working:
			# Run the benchmark for the proposed levels
			for levels in args.levelscore:
				# Check if level is still giving valid results
				if working[levels] == 0:
					continue
				# Set Core performance level to the one to be tested
				if editPerformanceLevel("core", int(levels)) == False:
					working[levels] = 0
					continue
				# Set Memory performance level to the highest
				if editPerformanceLevel("mem", 3) == False:
					working[levels] = 0
					continue
				# Get current DVFS settings - to make sure it was correctly applyed
				if currentPerfLevel() != (int(levels), 3):
					working[levels] = 0
					continue

				if args.v == 1:
					print("	Exploration of volt")

					# Run the benchmark
					commandBenchmark = benchmarkCommand(args.benchmark, folder, levels, 3, "CoreExploration", "Voltage")
					runBashCommand(commandBenchmark)

					# Write GPU Temp to end of output file
					appendCurrentTemp(commandBenchmark)
				else:
					print("	Exploration of freq")
		
					# Run the benchmark
					commandBenchmark = benchmarkCommand(args.benchmark, folder, levels, 3, "CoreExploration", "Frequency")
					runBashCommand(commandBenchmark)

					# Write GPU Temp to end of output file
					appendCurrentTemp(commandBenchmark)
					
			if args.v == 1:
				# Undervolt Core by 10mV
				CoreVoltage = [int(volt) - 10 for volt in CoreVoltage]
				for levels in range(0,8):
					if setPerformanceLevel("core", levels) == False:
						working[levels] = 0
			else
				# Overclock all levels Core by 10Hz
				CoreFrequency = [int(volt) + 10 for volt in CoreFrequency]
				for levels in range(0,8):
					if setPerformanceLevel("core", levels) == False:
						working[levels] = 0


# Exploration of Memory
elif args.m == 1:
	print("Exploration of Memory")
	working = [1] * 4
	while 1 in working:
		# Run the benchmark for the proposed levels
		for levels in args.levelsmemory:
			# Check if level is still giving valid results
			if working[levels] == 0:
				continue
			# Set Core performance level to the one to be tested
			if editPerformanceLevel("core", 7) == False:
				working[levels] = 0
				continue
			# Set Memory performance level to the highest
			if editPerformanceLevel("mem", int(levels)) == False:
				working[levels] = 0
				continue
			# Get current DVFS settings - to make sure it was correctly applyed
			if currentPerfLevel() != (7, int(levels)):
				working[levels] = 0
				continue

			if args.v == 1:
				print("	Exploration of volt")
				# Run the benchmark
				commandBenchmark = benchmarkCommand(args.benchmark, folder, 7, levels, "MemoryExploration", "Voltage")
				runBashCommand(commandBenchmark)

				# Write GPU Temp to end of output file
				appendCurrentTemp(commandBenchmark)
			else:
				# Run the benchmark
				commandBenchmark = benchmarkCommand(args.benchmark, folder, 7, levels, "MemoryExploration", "Frequency")
				runBashCommand(commandBenchmark)
				
				# Write GPU Temp to end of output file
				appendCurrentTemp(commandBenchmark)

		if args.v == 1:
			# Undervolt Memory by 10mV
			MemoryVoltage = [int(volt) - 10 for volt in MemoryVoltage]
			for levels in range(0,4):
				if setPerformanceLevel("mem", levels) == False:
					working[levels] = 0
		else:
			# Overclock Memory by 10Hz
			MemoryFreq = [int(freq) + 10 for freq in MemoryFreq]
			for levels in range(0,4):
				if setPerformanceLevel("mem", levels) == False:
					working[levels] = 0

else:
	print("No indication of exploration given [v:voltage, f:frequency]")
