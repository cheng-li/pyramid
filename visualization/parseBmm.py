import sys

pre_html = ''' <html>
    <head>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
        <script type="text/javascript" language="javascript" src="https://cdn.datatables.net/1.10.9/js/jquery.dataTables.min.js"></script>
        <script src="./js/jquery.min.js"></script>
        <script src="./js/jquery.dataTables.min.js"></script>
        <meta http-equiv="content-type" content="text/html; charset=utf-8" />
        
        <title>DataTables Editor - error</title>

        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/r/dt/jqc-1.11.3,dt-1.10.9,b-1.0.3,se-1.0.1/datatables.min.css">
        <link rel="stylesheet" type="text/css" href="css/generator-base.css">
        <link rel="stylesheet" type="text/css" href="css/editor.dataTables.min.css">

        <script type="text/javascript" charset="utf-8" src="https://cdn.datatables.net/r/dt/jqc-1.11.3,dt-1.10.9,b-1.0.3,se-1.0.1/datatables.min.js"></script>
        <script type="text/javascript" charset="utf-8" src="js/dataTables.editor.min.js"></script>
        <script type="text/javascript" charset="utf-8" src="js/table.error.js"></script>
        <style type="text/css">
        td.highlight {
            font-weight: bold;
            color: blue;
        }
        </style>
        <script type="text/javascript" class="init">
$(document).ready(function() {
'''

highlightValue = 0.1

def parseFile(fileName):
	output = open(fileName+".formated", "w")
	with open(fileName, 'r') as file:
		lines = file.readlines()
		numCluster = 0
		for line in lines:
			dictOut = {}
			if line.startswith("probabilities = "):
				line = line.split("[")[1]
				line = line.split("]")[0]
				elements = line.split(",")
				for ele in elements:
					singleValues = ele.split(":")
					key = singleValues[0] + "-" + singleValues[1]
					value = singleValues[-1]
					dictOut[key.strip()] = value.strip()

				numCluster += 1
				output.writelines("cluster: " + str(numCluster) + "\n")
				for key in sorted(dictOut):
					output.writelines(key + "\t" + dictOut[key] + "\n")
	file.close()
	output.close()

def parseFile2Dict(fileName):
	proportionList = []
	outputList = []
	with open(fileName, 'r') as file:
		lines = file.readlines()
		for line in lines:
			dictOut = {}
			if line.startswith("proportion = "):
				line = line.split("=")[1]
				prop = "%.4f" % float(line)
				proportionList.append(prop)
			if line.startswith("probabilities = "):
				line = line.split("[")[1]
				line = line.split("]")[0]
				elements = line.split(",")
				for ele in elements:
					singleValues = ele.split(":")
					key = singleValues[0] + "-" + singleValues[1]
					value = singleValues[-1]
					dictOut[key.strip()] = value.strip()
				outputList.append(dictOut)
				
	file.close()
	return proportionList, outputList

def transClusters2HTML():
	totalString = ""
	output = open("results.html", "w")
	clusterList = range(2, 13)
	# results = []
	# for i in clusterList:
	# 	inputFile = str(i)+".output"
	# 	results.append(parseFile2Dict(inputFile))

	loadData = ""
	for i in clusterList:
		loadData += "data"+str(i)+" = dataFromJson"+str(i)+"()\n"
	totalString = pre_html + loadData

	dataTable = ""
	for i in clusterList:
		dataTable += '''$('#table''' + str(i) + '''').DataTable( { \n '''
		dataTable += '''"aaData": data''' + str(i) + ",\n"
		dataTable += '''"iDisplayLength": 105,\n'''
		dataTable += '''"aoColumns": [\n'''
		dataTable += '''{ "mDataProp": "label" },\n'''
		for j in range(i):
			jj = j+1
			if (jj != i):
				dataTable += '''{ "mDataProp": "cluster''' + str(jj) + '''" },\n'''
			else:
				dataTable += '''{ "mDataProp": "cluster''' + str(jj) + '''" }\n'''
		dataTable += '''],\n'''
		dataTable += '''
		"createdRow": function ( row, data, index ) {
		var counterIndexCount = ''' + str(i) + "\n"
		dataTable += '''for (var i=1; i<=counterIndexCount; i++) {
                if (parseFloat(data["cluster"+i]) > ''' + str(highlightValue)
		dataTable += ''') {
                    $('td', row).eq([i]).addClass('highlight');
                }
            } 
            
            // }
        }'''

		dataTable += '''\n}); \n'''
	
	dataTable += '''});
		</script>
		</head>
		<body><br>
	'''

	totalString += dataTable


	htmlTable = ""
	for i in clusterList:
		inputFile = str(i)+".output"
		propResults, clusterResult = parseFile2Dict(inputFile)
		htmlTable += '''<h1>\n<span>Cluster: ''' + str(i) + '''</span>\n</h1>\n'''
		htmlTable += '''<table id="table''' + str(i) + '''" class="display" width="100%">
		<thead>
		<tr>
		<th>label</th>
		'''
		for j in range(i):
			jj = j+1
			htmlTable += '''<th>cluster'''+str(jj)+ '''(''' +str(propResults[j]) + ''')'''
			htmlTable += '''</th>\n'''
		htmlTable += '''</tr>
			</thead>
		</table>
		'''
	totalString += htmlTable


	loadFunction = ""
	loadFunction += '''<script>\n'''
	for i in clusterList:
		loadFunction += '''function dataFromJson'''+str(i) + '''() {\n'''
		loadFunction += '''return JSON.parse($('#json''' + str(i) +'''').html())\n}\n'''
	
	loadFunction += '''</script>\n'''
	totalString += loadFunction

	jsonString = ""
	for i in clusterList:
		inputFile = str(i)+".output"
		propResults, clusterResult = parseFile2Dict(inputFile)
		jsonString += '''<script id="json'''+str(i)+'''" type="application/json">\n'''
		jsonString += "["
		newline = ""
		for key in sorted(clusterResult[0]):
			newline += "{\"label\": \"" + key + "\", " 
			for j in range(len(clusterResult)):
				jj = j+1
				prob = "%.4f" % float(clusterResult[j][key])
				newline += "\"cluster" + str(jj) + "\": " + str(prob) + ", " 
			newline = newline[:-2]
			newline += "}, "
			
		newline = newline[:-2]
		jsonString += newline + ''']\n</script>\n'''

	totalString += jsonString + '''</script>
		</body></html>
		'''

	output.writelines(totalString)
	output.close()

if __name__ == "__main__":
	# parseFile(sys.argv[1])
	transClusters2HTML()

