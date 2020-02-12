import argparse
import subprocess
from pathlib import Path
import os
import re

CoreFreq = []
CoreVoltage = []
MemoryFreq = []
MemoryVoltage = []

def benchmarkCommand(benchmark, folder, levelCore, levelMem, typeEx, explore):
	global CoreFreq
	global CoreVoltage
	global MemoryFreq
	global MemoryVoltage
	name = benchmark.copy()
	name[0] = str(benchmark[0]).replace(str(folder) + '/', '')
	name = '_'.join(name)
	
	benchmark = ' '.join(benchmark)

	command = str(benchmark)
	file = str(folder) + "/Results/" + name + "-" + str(typeEx) + "-" + str(explore) + "-" + "Core-" + str(levelCore) + "-" + str(CoreFreq[levelCore]) + '-' + str(CoreVoltage[levelCore]) + "-Memory-" + str(levelMem) + '-' + str(MemoryFreq[levelMem]) + '-' + str(MemoryVoltage[levelMem]) + ".txt"

	return command, file

def runBashCommandOutputToFile(bashCommand, filePath, execution):
	"""Runs a bash command and outputs the process stdout and stderr to file

	Args:
		bashCommand: Command to be run
		filePath: Path of the output file

	Returns:
		The resulting process
	"""
	print("Running %s" % (bashCommand))
	output_file = open(filePath,'a')
	output_file.write("\n")
	output_file.write("#################################################\n")
	output_file.write("Execution: " + str(execution) + "\n")
	output_file.write("#################################################\n")
	output_file.write("\n")
	process = subprocess.run(bashCommand.split(), stdout=output_file, stderr=output_file, check=True, text=True)
	return process

def runBashCommand(bashCommand):
	"""Runs a bash command

	Args:
		bashCommand: Command to be run

	Returns:
		The resulting process
	"""
	print("Running %s" % (bashCommand))
	process = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
	return process

def runDVFSscript(command):
	"""Runs the DVFS script

	Args:
		command: Command to be run

	Returns:
		The resulting process
	"""
	script = "./DVFS " + str(command)
	process = runBashCommand(script)
	return process

def setPerformanceLevel(source, level):
	"""Sets a given performance level for the GPU Core and Memory.

	Args:
		source: string containing word "core" or "mem"
		level: an integer between 0-7 for core and 0-3 memory

	Returns:
		True - if action is sucessful.
		False - not possible to apply configuration.
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
		return False

	return True

def editPerformanceLevel(source, level, frequency, voltage):
	"""Edits a given performance level for the GPU Core and Memory.

	Args:
		source: string containing word "core" or "mem"
		level: an integer between 0-7 for core and 0-3 memory
		frequency: an integer indicating the frequency
		voltage: an integer indicating the voltage

	Returns:
		True - if action is sucessful.
		False - not possible to apply configuration.
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
		return False

	return True

def currentPerfLevel():
	"""Gets the current applied performance level for the Core and Memory

	Returns:
		Tuple on the form (core, memory) indicating the current 
		performance level of the two domains
	"""
	global CoreFreq
	global MemoryFreq
	result = runBashCommand("rocm-smi")
	core = -1
	mem = -1
	line = result.stdout.split('\n')
	line = line[5].split(" ")
	# Find indices of Core and Mem frequency
	indices = [i for i, s in enumerate(line) if 'Mhz' in s]
	core = line[indices[0]].replace("Mhz", '')
	mem = line[indices[1]].replace("Mhz", '')
	return CoreFreq.index(core), MemoryFreq.index(mem)

def appendCurrentTemp(file):
	result = runBashCommand("rocm-smi")
	line = result.stdout.split('\n')[5]
	# Find temperature
	temp = re.search("..\..c", line).group()

	print("File " , file)
	with open(file, "a+") as benchFile:
		benchFile.write("Temperature: " + str(temp.replace("c", " c")))

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
parser.add_argument('-t', '--tries', default=10, help="Number of times to perform the benchmark", type=int, choices=range(0, 51))
args = parser.parse_args()

if args.levelscore == None:
	args.levelscore = list(range(0,8))

if args.levelsmemory == None:
	args.levelsmemory = list(range(0,4))

if args.c == 1:
	print("Exploration of Core -", end =" ")
if args.m == 1:
	print("Exploration of Memory -", end =" ")
if args.v == 1:
	print("volt", end =" ")
if args.f == 1:
	print("frequency")
print()

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
	# Activates intended performance levels
	working_core = [0] * 8
	for i in args.levelscore:
		working_core[i] = [0] * 4
		for j in args.levelscore:
			working_core[i][j] = 1

	while 1 in working:
		# Run the benchmark for the proposed levels
		for levels in args.levelscore:
			# Check if level is still giving valid results
			if working[levels] == 0:
				continue
			# Set Core performance level to the one to be tested
			if setPerformanceLevel("core", int(levels)) == False:
				working[levels] = 0
				continue
			# Set Memory performance level to the highest
			if setPerformanceLevel("mem", 3) == False:
				working[levels] = 0
				continue
			# Get current DVFS settings - to make sure it was correctly applyed
			if currentPerfLevel() != (int(levels), 3):
				working[levels] = 0
				continue
			# Run the benchmark multiple times
			for i in range(0, args.tries):
				# Command to be launch
				if args.v == 1:
					commandBenchmark, fileBenchmark = benchmarkCommand(args.benchmark, folder, levels, 3, "CoreExploration", "Voltage")
				else:
					commandBenchmark, fileBenchmark = benchmarkCommand(args.benchmark, folder, levels, 3, "CoreExploration", "Frequency")

				# Run the benchmark
				runBashCommandOutputToFile(commandBenchmark, fileBenchmark, i)

				# Write GPU Temp to end of output file
				appendCurrentTemp(fileBenchmark)
				
		if args.v == 1:
			# Undervolt Core by 10mV
			CoreVoltage = [int(volt) - 10 for volt in CoreVoltage]
		else:
			# Overclock all levels Core by 10Hz
			CoreFrequency = [int(volt) + 10 for volt in CoreFrequency]

		# Apply new Power Table Settings
		for levels in range(0,8):
			if setPerformanceLevel("core", levels) == False:
				working[levels] = 0


# Exploration of Core
elif args.c == 1:
	# Activates intended performance levels
	working = [0] * 8
	for i in args.levelscore:
		working[i] = 1

	while 1 in working:
		# Run the benchmark for the proposed levels
		for levels in args.levelscore:
			# Check if level is still giving valid results
			if working[levels] == 0:
				continue
			# Set Core performance level to the one to be tested
			if setPerformanceLevel("core", int(levels)) == False:
				working[levels] = 0
				continue
			# Set Memory performance level to the highest
			if setPerformanceLevel("mem", 3) == False:
				working[levels] = 0
				continue
			# Get current DVFS settings - to make sure it was correctly applyed
			if currentPerfLevel() != (int(levels), 3):
				working[levels] = 0
				continue
			# Run the benchmark multiple times
			for i in range(0, args.tries):
				# Command to be launch
				if args.v == 1:
					commandBenchmark, fileBenchmark = benchmarkCommand(args.benchmark, folder, levels, 3, "CoreExploration", "Voltage")
				else:
					commandBenchmark, fileBenchmark = benchmarkCommand(args.benchmark, folder, levels, 3, "CoreExploration", "Frequency")

				# Run the benchmark
				runBashCommandOutputToFile(commandBenchmark, fileBenchmark, i)

				# Write GPU Temp to end of output file
				appendCurrentTemp(fileBenchmark)
				
		if args.v == 1:
			# Undervolt Core by 10mV
			CoreVoltage = [int(volt) - 10 for volt in CoreVoltage]
		else:
			# Overclock all levels Core by 10Hz
			CoreFrequency = [int(volt) + 10 for volt in CoreFrequency]

		# Apply new Power Table Settings
		for levels in range(0,8):
			if editPerformanceLevel("core", levels, CoreFrequency[levels], CoreVoltage[levels]) == False:
				working[levels] = 0


# Exploration of Memory
elif args.m == 1:
	# Activates intended performance levels
	working = [0] * 4
	for i in args.levelsmemory:
		working[i] = 1

	while 1 in working:
		# Run the benchmark for the proposed levels
		for levels in args.levelsmemory:
			print("levels " + str(levels))
			# Check if level is still giving valid results
			if working[levels] == 0:
				continue
			# Set Core performance level to the one to be tested
			if setPerformanceLevel("core", 7) == False:
				print("	Not able to select core level.")
				working[levels] = 0
				continue
			# Set Memory performance level to the highest
			if setPerformanceLevel("mem", int(levels)) == False:
				print("	Not able to select memory level.")
				working[levels] = 0
				continue
			# Get current DVFS settings - to make sure it was correctly applyed
			cur = currentPerfLevel()
			if cur  != (7, int(levels)):
				print("	Selected Performance Levels don't match current ones.")
				print("	", end=" ")
				print(cur, end=" ")
				print("!= (7, " + str(levels) + ")")
				working[levels] = 0
				continue

			# Run the benchmark multiple times
			for i in range(0, args.tries):
				# Command to be launch
				if args.v == 1:
					commandBenchmark, fileBenchmark = benchmarkCommand(args.benchmark, folder, 7, levels, "MemoryExploration", "Voltage")
				else:
					commandBenchmark, fileBenchmark = benchmarkCommand(args.benchmark, folder, 7, levels, "MemoryExploration", "Frequency")

				# Run the benchmark
				runBashCommandOutputToFile(commandBenchmark, fileBenchmark, i)
				# Write GPU Temp to end of output file
				appendCurrentTemp(fileBenchmark)

		if args.v == 1:
			# Undervolt Memory by 10mV
			MemoryVoltage = [int(volt) - 10 for volt in MemoryVoltage]
		else:
			# Overclock Memory by 10Hz
			MemoryFreq = [int(freq) + 10 for freq in MemoryFreq]

		# Apply new Power Table Settings
		for levels in range(0,4):
			if editPerformanceLevel("mem", levels, MemoryFreq[levels], MemoryVoltage[levels]) == False:
				working[levels] = 0

else:
	print("No indication of exploration given [v:voltage, f:frequency]")

# GPU fan automatic
result = runBashCommand("rocm-smi --resetfans")
if "Successfully" not in result.stdout:
	print("Not able to set fan")
	exit()