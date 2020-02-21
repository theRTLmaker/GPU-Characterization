import argparse
import re
import glob
import pandas as pd
import numpy as np

# import xlsxwriter as xl
# import numpy as  np
# from vincent.colors import brews

# Default DVFS values
defaultCore = [(852, 800), (991, 900), (1138, 950), (1269, 1000), (1348, 1050),
               (1440, 1100), (1528, 1150), (1601, 1200)]
defaultMemory = [(167, 800), (500, 900), (800, 950), (946, 1000)]


def colnum_string(n):
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string

def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list


# Parser to collect user given arguments
parser = argparse.ArgumentParser(
    prog="analysisBenchmark",
    description='Extract the results from the output files.')
parser.add_argument('-p',
                    '--path',
                    metavar='path',
                    type=str,
                    help="Path to folder results",
                    required=True,
                    nargs='+')
parser.add_argument('-c',
                    '--config',
                    metavar='path',
                    type=str,
                    help="Config file",
                    required=True,
                    nargs='+')
args = parser.parse_args()

# List of entries to find on the output file
expectedOutput = []

# Parse config file
config_file = open(" ".join(args.config), "r")
while True:
    line = config_file.readline()
    if not line:
        break

    # Check the type of content of the output file
    parseType = re.search(r"(.*?)\=", line)
    if parseType is not None:
        # Get the information written on the name of the file
        if parseType.group(0) == "FILE=":
            fileConvention = str(args.path[0]) + str(config_file.readline())
        # Get the expected output lines
        elif parseType.group(0) == "OUTPUT=":
            while True:
                outputLine = config_file.readline()
                if not outputLine:
                    break
                if re.search(r"\{(.*?)\}", outputLine) is not None:
                    expectedOutput.append(outputLine)
            break

# List of content on the file name
fileNameVars = re.findall(r"\{(.*?)\}", fileConvention) + [
    'target', 'object of study', 'core performance level', 'core frequency',
    'core voltage', 'memory performance level', 'memory frequency',
    'memory voltage'
]

# Create the regular expressions patterns to get the data from the filename
while True:
    match = re.search(r"\{(.*?)\}", fileConvention)
    if match is None:
        break
    fileConvention = fileConvention[0:match.span()[0]:] + "(.*?)" + fileConvention[match.span()[1]::]

# Name of the variables to extract from the file output
outputNameVars = []
# Additional information for post processing data analysis
varAnalysis = {}
for idx, line in enumerate(expectedOutput):
    match = re.search(r"\{(.*?)\}", line)
    expectedOutput[idx] = line[0:match.span()[0]:] + "(.*?)" + line[match.span()[1]::]
    outputNameVars.append(re.sub(r'\([^)]*\)\)', '', match.groups()[0]))
    match = re.search(r"\((.*?)\((.*?)\)\)", match.groups()[0])
    if match is not None:
        match = match.groups()
        varAnalysis[outputNameVars[-1]] = [str(match[0]).replace(" ", "").split(","), str(match[1])]

# Get all the files on the results folder
files = [f for f in glob.glob(args.path[0] + "/*.txt", recursive=True)]

# Dictionary holding a pandas dataframe for every benchmark type
Benchmark = {}

# Number of executions of every benchmark
numberOfExecutions = 0

# Run throw all the output files
for f in files:
    # Get the benchmark type
    regex = re.compile(fileConvention[:-1] + "-")
    benchmarkType = []
    if regex.match(f).groups():
        for i in regex.match(f).groups():
            benchmarkType.append(i)
        benchmarkType = "-".join(benchmarkType)
    else:
        benchmarkType = 'Data'

    # Get the data from the filename
    regex = re.compile(
        fileConvention[:-1] +
        "-(.*?)-(.*?)-Core-(.*?)-(.*?)-(.*?)-Memory-(.*?)-(.*?)-(.*?).txt")
    fileNameValues = []
    for i in regex.match(f).groups():
        try:
            fileNameValues.append(int(i))
        except:
            fileNameValues.append(i)
    params = dict(zip(fileNameVars, fileNameValues))

    # Open the file and search for the output content
    with open(f, "r") as search:
        for idx, regex in enumerate(expectedOutput):
            numberOfExecutions = 0
            regexx = re.compile(regex)
            for line in search:
                value = regexx.search(line)
                if value is not None:
                    value = value.group()
                    for word in regex.split(" "):
                        value = value.replace(word, "")
                    try:
                        params[outputNameVars[idx] + " " +
                               str(numberOfExecutions)] = float(value)
                    except:
                        value = value.replace("\n", "")
                        params[outputNameVars[idx] + " " +
                               str(numberOfExecutions)] = value.replace(
                                   " ", "")
                    numberOfExecutions = numberOfExecutions + 1
            search.seek(0)

    try:
        if int(benchmarkType) not in Benchmark:
            Benchmark[int(benchmarkType)] = []
        Benchmark[int(benchmarkType)].append(params)
    except:
        continue
        if benchmarkType not in Benchmark:
            Benchmark[benchmarkType] = []
        Benchmark[benchmarkType].append(params)

# Sort the benchmark types ascending
temp = Benchmark.copy()
Benchmark = {}
for key in sorted(temp):
    Benchmark[key] = temp[key]

# Order the of the dataframe collumns
order = [
    'target', 'object of study', 'core performance level', 'core frequency',
    'core voltage', 'memory performance level', 'memory frequency',
    'memory voltage'
]
for value in outputNameVars:
    for i in range(numberOfExecutions):
        order.append(value + " " + str(i))

# Create a Pandas Excel writer using XlsxWriter as the engine.
excel_file = "./" + str(args.path[0]) + 'results.xlsx'
sheet_name = 'Data'

writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')

# Create a pandas dataframe for every benchmark type
# Sort the values by performance level, frequency and voltage for the core and memory
Benchmark_dt = {}
for key, value in Benchmark.items():
    Benchmark_dt[key] = pd.DataFrame.from_dict(Benchmark[key])
    Benchmark_dt[key] = Benchmark_dt[key][order]
    Benchmark_dt[key].sort_values(
        by=[
            'target', 'object of study', 'core performance level',
            'core frequency', 'core voltage', 'memory performance level',
            'memory frequency', 'memory voltage'
        ],
        ascending=[True, True, True, False, False, True, False, False],
        inplace=True)
    Benchmark_dt[key].dropna(inplace=True)
    Benchmark_dt[key].set_index([
        'target', 'object of study', 'core performance level',
        'core frequency', 'core voltage', 'memory performance level',
        'memory frequency', 'memory voltage'
    ],
                                inplace=True)


    # Outliers removal
    # Go through every row of the dataframe and remove all the values that are on the 5% bigger and smaller
    for index, row in Benchmark_dt[key].iterrows():
        # Check for every variable that is going to suffer analysis, what are the outliers
        # Everytime that one experiment is selected as outlier for one variable, that experiment is removed
        # From all variables
        mask = None
        for var, analysisList in varAnalysis.items():
            # Gets the collums name of collumns containing the general name in var
            cols = [
                col for col in Benchmark_dt[key]
                if var in col and not any(sb in col for sb in [
                    "average", "median", "min", "max", "mode", "boolean",
                    "delta"
                ])
            ]            
            if mask is not None:
                mask = mask.values & row[cols].between(row[cols].quantile(.05), row[cols].quantile(.95)).values
            else:
                mask = row[cols].between(row[cols].quantile(.05), row[cols].quantile(.95))
        notMask = [not boolean for boolean in mask]
        # Remove the invalid run from all types of data collected for that row
        for var in outputNameVars:
            # Gets the collums name of collumns containing the general name in var
            cols = [
                col for col in Benchmark_dt[key]
                if var in col and not any(sb in col for sb in [
                    "average", "median", "min", "max", "mode", "boolean",
                    "delta"
                ])
            ]
            Benchmark_dt[key].at[index, cols] = row[cols].where(mask, other=np.NaN)
            
    # Compute data analysis collumns
    for var, analysisList in varAnalysis.items():
        # Remove outliers
        # Gets the collums name of collumns containing the general name in var
        cols = [
            col for col in Benchmark_dt[key]
            if var in col and not any(sb in col for sb in [
                "average", "median", "min", "max", "mode", "boolean",
                "delta"
            ])
        ]
        for analysis in analysisList[0]:
            if analysis == "average":
                Benchmark_dt[key][
                    str(var) + " " +
                    str(analysis)] = Benchmark_dt[key][cols].mean(axis=1)
            elif analysis == "median":
                Benchmark_dt[key][
                    str(var) + " " +
                    str(analysis)] = Benchmark_dt[key][cols].median(axis=1)
            elif analysis == "min":
                Benchmark_dt[key][str(var) + " " +
                                  str(analysis)] = Benchmark_dt[key][cols].min(
                                      axis=1)
            elif analysis == "max":
                Benchmark_dt[key][str(var) + " " +
                                  str(analysis)] = Benchmark_dt[key][cols].max(
                                      axis=1)
            elif analysis == "mode":
                Benchmark_dt[key][
                    str(var) + " " +
                    str(analysis)] = Benchmark_dt[key][cols].mode(axis=1)
            elif analysis == "boolean":
                continue
            # Compute the delta
            for index, row in Benchmark_dt[key].iterrows():
                pos = (index[0], index[1], index[2], defaultCore[index[2]][0],
                       defaultCore[index[2]][1], index[5],
                       defaultMemory[index[5]][0], defaultMemory[index[5]][1])

                Benchmark_dt[key].at[
                    index, "delta " + str(var) + " " + str(analysis)] = (
                        Benchmark_dt[key].at[index,
                                             str(var) + " " + str(analysis)] -
                        Benchmark_dt[key].at[pos,
                                             str(var) + " " + str(analysis)]
                    ) / Benchmark_dt[key].at[pos,
                                             str(var) + " " +
                                             str(analysis)] * 100

# Count the number of entries per level
# numberOfEntries = np.zeros(8)
# for i in range(0,8):
#   numberOfEntries[i] = len(result_dt[result_dt['LEVEL'] == i])

# Write the values to the excel file
for key, value in Benchmark.items():
    # Count the number of different DVFS performance level existent on the files
    performancePairs = {}
    for index in Benchmark_dt[key].index.values:
        pair = (index[2], index[5])
        if pair not in performancePairs:
            performancePairs[pair] = 1
        else:
            performancePairs[pair] += 1

    # Translate the dataframe to excel
    Benchmark_dt[key].to_excel(writer, sheet_name=str(key))

    workbook = writer.book
    worksheet = writer.sheets[str(key)]

    # Get the number of experiments done
    startGraphs = Benchmark_dt[key].count().max() + 1
    i = 0
    # Run over the variable analysis to be computed
    for var, analysisList in varAnalysis.items():
        # Run over the types of analysis to performed
        for analysis in analysisList[0]:
            j = 0
            totalEntriesPerPair = 1
            # Run over the different performance DVFS domains existent
            for pair, numberOfEntries in performancePairs.items():
                # Create the raw data graph and the delta one
                for dataType in ['', 'delta ']:
                    # Get the horizontal position of the data on the Excel file
                    columnLetters = colnum_string(
                        Benchmark_dt[key].columns.get_loc(str(dataType) + str(var) + " " + str(analysis)) + 9)

                    # Create a chart to represent the total training time
                    chart = workbook.add_chart({
                        'type': 'scatter',
                        'subtype': 'straight_with_markers'
                    })

                    # Configure the series of the chart from the dataframe data.
                    char_data = {
                        # TODO meter no ficheiro de config o tipo de dados - titulo e eixos
                        'name': dataType + 'Time',
                        'values':'=' + str(key) + '!$' + columnLetters + '$' + str(totalEntriesPerPair + 1) +':$' + columnLetters + '$' + str(totalEntriesPerPair + numberOfEntries),
                    }

                    if Benchmark_dt[key].index.values[0][0] == "MemoryExploration":
                        if Benchmark_dt[key].index.values[0][1] == "Voltage":
                            char_data['categories'] = '=' + str(key) + '!$H$' + str(totalEntriesPerPair + 1) + ':$H$' + str(totalEntriesPerPair + numberOfEntries)
                            chart.set_x_axis({
                                'name': 'Voltage [mV]',
                                'min': 800,
                                'max': 1200
                            })
                        else:
                            char_data['categories'] = '=' + str(key) + '!$G$' + str(totalEntriesPerPair + 1) + ':$G$' + str(totalEntriesPerPair + numberOfEntries)
                            chart.set_x_axis({
                                'name': 'Frequency [Hz]',
                                'min': 800,
                                'max': 1600
                            })
                    elif Benchmark_dt[key].index.values[0][0] == "CoreExploration":
                        if Benchmark_dt[key].index.values[0][1] == "Voltage":
                            char_data['categories'] = '=' + str(key) + '!$E$' + str(totalEntriesPerPair + 1) + ':$E$' + str(totalEntriesPerPair + numberOfEntries)
                            chart.set_x_axis({
                                'name': 'Voltage [mV]',
                                'min': 800,
                                'max': 1200
                            })
                        else:
                            char_data['categories'] = '=' + str(key) + '!$D$' + str(totalEntriesPerPair + 1) + ':$D$' + str(totalEntriesPerPair + numberOfEntries)
                            chart.set_x_axis({
                                'name': 'Frequency [Hz]',
                                'min': 800,
                                'max': 1600
                            })

                    chart.add_series(char_data)
                    # Configure the chart axes.
                    if "delta" in dataType:
                    	chart.set_title({'name': str(dataType) + str(var) + " [%]"})
                    	chart.set_y_axis({'name': str(dataType) + " [%]", 'major_gridlines': {'visible': True}})
                    else:
                    	chart.set_title({'name': str(var) + " [" + str(analysis) + "]"})
                    	chart.set_y_axis({'name': str(dataType) + " " + str(analysisList[1]), 'major_gridlines': {'visible': True}})


                    # Insert the chart into the worksheet.
                    worksheet.insert_chart(
                        'A' + str(startGraphs + 2), chart, {
                            'x_offset': i * 500,
                            'y_offset': j * 300,
                            'x_scale': 1,
                            'y_scale': 1
                        })
                    j += 1
                totalEntriesPerPair += numberOfEntries
            i += 1
writer.save()

# Access the XlsxWriter workbook and worksheet objects from the dataframe.

# actual_line = 2

# # Run all performance values
# for i in range(0, 8):
#     nameF = result_dt.iloc[actual_line - 2, 1]

#     ########### TIME CHART ################

#     # Create a chart to represent the total training time
#     time_chart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     # Configure the series of the chart from the dataframe data.
#     time_chart.add_series({
#         'name':
#         'LEVEL ' + str(i) + ', FREQ ' + str(nameF) + ' MHz',
#         'values':
#         '=' + str(sheet_name) + '!$BL$' + str(actual_line) + ':$BL$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#         'categories':
#         '=' + str(sheet_name) + '!$D$' + str(actual_line) + ':$D$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#     })

#     # Configure the chart axes.
#     time_chart.set_title({'name': 'Time to Train:'})
#     time_chart.set_x_axis({'name': 'Voltage [mV]', 'min': 800, 'max': 1200})
#     time_chart.set_y_axis({
#         'name': 'Time [s]',
#         'major_gridlines': {
#             'visible': True
#         },
#         'min': 256,
#         'max': 270
#     })
#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A280', time_chart, {
#         'x_offset': i * 750,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     ########### Delta TIME CHART ################

#     # Create a chart to represent the total training time
#     time_chart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     # Configure the series of the chart from the dataframe data.
#     time_chart.add_series({
#         'name':
#         'LEVEL ' + str(i) + ', FREQ ' + str(nameF) + ' MHz',
#         'values':
#         '=' + str(sheet_name) + '!$EI$' + str(actual_line) + ':$EI$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#         'categories':
#         '=' + str(sheet_name) + '!$D$' + str(actual_line) + ':$D$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#     })

#     # Configure the chart axes.
#     time_chart.set_title({'name': 'Delta Time to Train:'})
#     time_chart.set_x_axis({'name': 'Voltage [mV]', 'min': 800, 'max': 1200})
#     time_chart.set_y_axis({
#         'name': 'Delta Time [%]',
#         'major_gridlines': {
#             'visible': True
#         },
#         'min': -4,
#         'max': 3
#     })
#     #
#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A303', time_chart, {
#         'x_offset': i * 750,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     ########### Power CHART ################

#     # Create a chart object.
#     power_chart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     # Configure the series of the chart from the dataframe data.
#     power_chart.add_series({
#         'name':
#         'MAX POWER',
#         'values':
#         '=' + str(sheet_name) + '!$BN$' + str(actual_line) + ':$BN$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#         'categories':
#         '=' + str(sheet_name) + '!$D$' + str(actual_line) + ':$D$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#     })

#     # Configure the series of the chart from the dataframe data.
#     power_chart.add_series({
#         'name':
#         'AVG POWER',
#         'values':
#         '=' + str(sheet_name) + '!$BM$' + str(actual_line) + ':$BM$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#         'categories':
#         '=' + str(sheet_name) + '!$D$' + str(actual_line) + ':$D$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#     })

#     # Configure the chart axes.
#     power_chart.set_title({'name': 'MAX and AVG POWER'})
#     power_chart.set_x_axis({'name': 'Voltage [mV]', 'min': 800, 'max': 1200})
#     power_chart.set_y_axis({
#         'name': 'Power [W]',
#         'major_gridlines': {
#             'visible': True
#         },
#         'min': 0,
#         'max': 120
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A187', power_chart, {
#         'x_offset': i * 750,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     ########### Delta MAX and AVG Power CHART ################

#     # Create a chart object.
#     power_chart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     # Configure the series of the chart from the dataframe data.
#     power_chart.add_series({
#         'name':
#         'MAX POWER',
#         'values':
#         '=' + str(sheet_name) + '!$EF$' + str(actual_line) + ':$EF$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#         'categories':
#         '=' + str(sheet_name) + '!$D$' + str(actual_line) + ':$D$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#     })

#     # Configure the series of the chart from the dataframe data.
#     power_chart.add_series({
#         'name':
#         'AVG POWER',
#         'values':
#         '=' + str(sheet_name) + '!$EG$' + str(actual_line) + ':$EG$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#         'categories':
#         '=' + str(sheet_name) + '!$D$' + str(actual_line) + ':$D$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#     })

#     # Configure the chart axes.
#     power_chart.set_title({'name': 'Delta MAX and AVG POWER'})
#     power_chart.set_x_axis({'name': 'Voltage [mV]', 'min': 800, 'max': 1200})
#     power_chart.set_y_axis({
#         'name': 'Power [W]',
#         'major_gridlines': {
#             'visible': True
#         },
#         'min': -60,
#         'max': 80
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A210', power_chart, {
#         'x_offset': i * 750,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     ########### ENERGY CHART ################

#     # Create a chart object.
#     energy_chart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     # Configure the series of the chart from the dataframe data.
#     energy_chart.add_series({
#         'name':
#         'ENERGY',
#         'values':
#         '=' + str(sheet_name) + '!$BO$' + str(actual_line) + ':$BO$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#         'categories':
#         '=' + str(sheet_name) + '!$D$' + str(actual_line) + ':$D$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#     })

#     # Configure the chart axes.
#     energy_chart.set_title({'name': 'ENERGY'})
#     energy_chart.set_x_axis({'name': 'Voltage [mV]', 'min': 800, 'max': 1200})
#     energy_chart.set_y_axis({
#         'name': 'Energy [J]',
#         'major_gridlines': {
#             'visible': True
#         },
#         'min': 3500,
#         'max': 15000
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A233', energy_chart, {
#         'x_offset': i * 750,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     ########### Delta ENERGY CHART ################

#     # Create a chart object.
#     energy_chart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     # Configure the series of the chart from the dataframe data.
#     energy_chart.add_series({
#         'name':
#         'ENERGY',
#         'values':
#         '=' + str(sheet_name) + '!$EH$' + str(actual_line) + ':$EH$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#         'categories':
#         '=' + str(sheet_name) + '!$D$' + str(actual_line) + ':$D$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#     })

#     # Configure the chart axes.
#     energy_chart.set_title({'name': 'Delta ENERGY'})
#     energy_chart.set_x_axis({'name': 'Voltage [mV]', 'min': 800, 'max': 1200})
#     energy_chart.set_y_axis({
#         'name': 'Delta Energy [%]',
#         'major_gridlines': {
#             'visible': True
#         },
#         'min': -30,
#         'max': 30
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A256', energy_chart, {
#         'x_offset': i * 750,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     ########### Delta Accuracy CHART ################

#     # Create a chart object.
#     chart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     # Configure the series of the chart from the dataframe data.
#     chart.add_series({
#         'name':
#         '=' + str(sheet_name) + '!$D$' + str(actual_line),
#         'values':
#         '=' + str(sheet_name) + '!$EE$' + str(actual_line) + ':$EE$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#         'categories':
#         '=' + str(sheet_name) + '!$D$' + str(actual_line) + ':$D$' +
#         str(actual_line + int(numberOfEntries[i]) - 1),
#     })

#     # Configure the chart axes.
#     chart.set_title({'name': 'Delta Accuracy'})
#     chart.set_x_axis({'name': 'Voltage [mV]', 'min': 800, 'max': 1200})
#     chart.set_y_axis({
#         'name': 'Delta Accuracy [%]',
#         'major_gridlines': {
#             'visible': True
#         },
#         'min': -0.1,
#         'max': 0.1
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A165', chart, {
#         'x_offset': i * 750,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     ########### Accuracy CHART ################

#     # Create a chart object.
#     chart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     # Configure the series of the chart from the dataframe data.
#     for a in range(int(numberOfEntries[i])):
#         chart.add_series({
#             'name':
#             '=' + str(sheet_name) + '!$D$' + str(actual_line),
#             'values':
#             '=' + str(sheet_name) + '!$E$' + str(actual_line) + ':$X$' +
#             str(actual_line),
#             'categories':
#             '=' + str(sheet_name) + '!$AS$' + str(actual_line) + ':$BL$' +
#             str(actual_line),
#         })
#         actual_line += 1

#     # Configure the chart axes.
#     chart.set_title(
#         {'name': 'LEVEL ' + str(i) + ', FREQ ' + str(nameF) + ' MHz'})
#     chart.set_x_axis({'name': 'Time [s]'})
#     chart.set_y_axis({
#         'name': 'Accuracy [%]',
#         'major_gridlines': {
#             'visible': True
#         },
#         'min': 0,
#         'max': 100
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A142', chart, {
#         'x_offset': i * 750,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

# actual_line = 2

# for i in range(0, 8):
#     nameF = result_dt.iloc[actual_line - 2, 1]
#     worksheet = workbook.add_worksheet("Level" + str(i))
#     for j in range(0, 8):
#         worksheet.write(0, j, j)
#     ########### Accuracy CHART ################

#     # Create a chart object.
#     chart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     delta_accuracy = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     ########### Power CHART ################

#     # Create a chart object.
#     avgpowerchart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     # Create a chart object.
#     maxpowerchart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     ########### Energy CHART ################

#     # Create a chart object.
#     enerychart = workbook.add_chart({
#         'type': 'scatter',
#         'subtype': 'smooth_with_markers'
#     })

#     # Configure the series of the chart from the dataframe data.
#     for a in range(int(numberOfEntries[i])):
#         chart.add_series({
#             'name':
#             '=' + str(sheet_name) + '!$D$' + str(actual_line),
#             'values':
#             '=' + str(sheet_name) + '!$CA$' + str(actual_line) + ':$CH$' +
#             str(actual_line),
#             'categories':
#             '=Level' + str(i) + '!$A$1:$H$1'
#         })

#         delta_accuracy.add_series({
#             'name':
#             '=' + str(sheet_name) + '!$D$' + str(actual_line),
#             'values':
#             '=' + str(sheet_name) + '!$EJ$' + str(actual_line) + ':$EQ$' +
#             str(actual_line),
#             'categories':
#             '=Level' + str(i) + '!$A$1:$H$1'
#         })

#         avgpowerchart.add_series({
#             'name':
#             '=' + str(sheet_name) + '!$D$' + str(actual_line),
#             'values':
#             '=' + str(sheet_name) + '!$CQ$' + str(actual_line) + ':$CX$' +
#             str(actual_line),
#             'categories':
#             '=Level' + str(i) + '!$A$1:$H$1'
#         })
#         maxpowerchart.add_series({
#             'name':
#             '=' + str(sheet_name) + '!$D$' + str(actual_line),
#             'values':
#             '=' + str(sheet_name) + '!$CY$' + str(actual_line) + ':$DF$' +
#             str(actual_line),
#             'categories':
#             '=Level' + str(i) + '!$A$1:$H$1'
#         })

#         enerychart.add_series({
#             'name':
#             '=' + str(sheet_name) + '!$D$' + str(actual_line),
#             'values':
#             '=' + str(sheet_name) + '!$DG$' + str(actual_line) + ':$DN$' +
#             str(actual_line),
#             'categories':
#             '=Level' + str(i) + '!$A$1:$H$1'
#         })
#         actual_line += 1

#     # Configure the chart axes.
#     chart.set_title(
#         {'name': 'LEVEL ' + str(i) + ', FREQ ' + str(nameF) + ' MHz'})
#     chart.set_x_axis({'name': 'Performance Level', 'min': 0, 'max': 7})
#     chart.set_y_axis({
#         'name': 'Accuracy [%]',
#         'major_gridlines': {
#             'visible': True
#         },
#         'min': 98,
#         'max': 100
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A1', chart, {
#         'x_offset': 0,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     # Configure the chart axes.
#     delta_accuracy.set_title({'name': 'Delta Accuracy'})
#     delta_accuracy.set_x_axis({
#         'name': 'Performance Level',
#         'min': 0,
#         'max': 7
#     })
#     delta_accuracy.set_y_axis({
#         'name': 'Delta Accuracy [%]',
#         'major_gridlines': {
#             'visible': True
#         },
#         'min': -0.1,
#         'max': 0.1
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('M1', delta_accuracy, {
#         'x_offset': 0,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     # Configure the chart axes.
#     avgpowerchart.set_title({'name': 'AVG POWER'})
#     avgpowerchart.set_x_axis({'name': 'Performance Level', 'min': 0, 'max': 7})
#     avgpowerchart.set_y_axis({
#         'name': 'Power [W]',
#         'major_gridlines': {
#             'visible': True
#         }
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A24', avgpowerchart, {
#         'x_offset': 0,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     # Configure the chart axes.
#     maxpowerchart.set_title({'name': 'MAX POWER}'})
#     maxpowerchart.set_x_axis({'name': 'Performance Level', 'min': 0, 'max': 7})
#     maxpowerchart.set_y_axis({
#         'name': 'Power [W]',
#         'major_gridlines': {
#             'visible': True
#         }
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A47', maxpowerchart, {
#         'x_offset': 0,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

#     # Configure the chart axes.
#     enerychart.set_title({'name': 'Energy'})
#     enerychart.set_x_axis({'name': 'Performance Level', 'min': 0, 'max': 7})
#     enerychart.set_y_axis({
#         'name': 'Energy [J]',
#         'major_gridlines': {
#             'visible': True
#         }
#     })

#     # Insert the chart into the worksheet.
#     worksheet.insert_chart('A70', enerychart, {
#         'x_offset': 0,
#         'y_offset': 0,
#         'x_scale': 1.5,
#         'y_scale': 1.5
#     })

# # Close the Pandas Excel writer and output the Excel file.
# writer.save()
