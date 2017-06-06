#!/usr/bin/env python
from elasticsearch import Elasticsearch
import json
import time
import sys, getopt
import re
import os
from os import listdir
from os.path import isfile, join
# from sets import Set


# This program read in a json file data and output to a html file.
# The data is then shown in a big table with many options in front of the page.
# The javascript then adjust the style(block or none) of each cell to determine
# whether to display the cells in the table.


def writeHeader(output):
    output.write(header)


def getPositions(docId, field, keywords, slop, in_order):
    clauses = []
    for keyword in keywords.split():
        clauses.append({'span_term':{ field : keyword}})
    res = es.search(index=esIndex,
                    doc_type="document",
                    body={"explain":False,
                          "query":{
                              "filtered":{
                                  "query":{
                                      "span_near": {
                                          "clauses": clauses,
                                          "slop": slop,
                                          "in_order": in_order,
                                          "collect_payloads": False}},
                                  "filter":{"ids":{"values":[docId]}}}},
                          "highlight":{"fields":{field:{}}},
                          "size":1})
    if len(res["hits"]["hits"]) <= 0:
        return []
    positions = []

    for field in res["hits"]["hits"][0]["highlight"]:
        text = res["hits"]["hits"][0]["_source"][field]
        highlights = res["hits"]["hits"][0]["highlight"][field]
        for HL in highlights:
            cleanHL = HL.replace("<em>", "")
            cleanHL = cleanHL.replace("</em>", "")
            baseindex = text.find(cleanHL)

            # in case the highlight not found in body
            if baseindex == -1:
                continue

            while HL.find("<em>") != -1:
                start = HL.find("<em>") + baseindex
                HL = HL.replace("<em>", "", 1)
                end = HL.find("</em>") + baseindex
                HL = HL.replace("</em>", "", 1)
                curPos = [start, end]
                positions.append(curPos)
    return positions


def writeRule(docId, line_count, num, rule):
    oneRule = {}

    oneRule['score'] = rule['score']
    oneRule['checks'] = []
    
    # BW: add all position into one list and field
    allPos = []
    field = ""
    for check in rule["checks"]:
        #read pos from ElasticSearch
        checkOneRule = {}
        pos = []
        if check["feature value"] != 0.0 and check["feature"].has_key("ngram"):
            pos = getPositions(docId, check["feature"]["field"], check["feature"]["ngram"], 
                check["feature"]["slop"], check["feature"]["inOrder"])
        
        if not check["feature"].has_key("ngram"):
            checkOneRule["name"] = check["feature"]["name"]
            checkOneRule["index"] = check["feature"]["index"]
        else:
            checkOneRule["ngram"] = check["feature"]["ngram"]
            checkOneRule["index"] = check["feature"]["index"]
            checkOneRule["field"] = check["feature"]["field"]
            checkOneRule["slop"] = check["feature"]["slop"]
            field = checkOneRule["field"]

        checkOneRule['value'] = check["feature value"]
        checkOneRule['relation'] = check["relation"]
        checkOneRule['threshold'] = check["threshold"]
        checkOneRule['highlights'] = str(pos)
        allPos.extend(pos)

        oneRule['checks'].append(checkOneRule)
        
    return oneRule, allPos, field


def writeClass(docId, line_count, clas, classDescription):
    oneClass = {}
    oneClass['id'] = clas["internalClassIndex"]
    ## add classDescription
    name = clas["className"]
    if name in classDescription:
        name = name + " : " + classDescription[name]
    else:
        name = name + " : " + "MISSING DESCRIPTION"
    oneClass['name'] = name
    oneClass['classProbability'] = clas["classProbability"]
    oneClass['totalScore'] = clas["classScore"]

    # default rule number is 6
    start = 0
    if not clas["rules"][0].has_key("checks"):
        oneClass['prior'] = clas["rules"][0]["score"]
        start = 1

    oneClass['rules'] = []
    allPos = []
    field = ""
    for i in range(start, len(clas["rules"])):
        oneRule, pos, field = writeRule(docId, line_count, i, clas["rules"][i])
        oneClass['rules'].append(oneRule)
        allPos.extend(pos)

    oneClass['allPos'] = str(allPos)
    oneClass['field'] = field

    return oneClass


def createTFPNColumns(row, line_count, oneRow, classDescription):
    tmpDict = []
    # build set
    labelSet = set()

    # column 4 TP
    oneRow['TP'] = []
    for clas in row["classScoreCalculations"]:
        if clas['internalClassIndex'] in row["internalLabels"] and clas['internalClassIndex'] in row["internalPrediction"]:
            oneRow['TP'].append(writeClass(row["id"], line_count, clas, classDescription))

    # column 5 FP
    oneRow['FP'] = []
    for clas in row["classScoreCalculations"]:
        if clas['internalClassIndex'] not in row["internalLabels"] and clas['internalClassIndex'] in row["internalPrediction"]:
            oneRow['FP'].append(writeClass(row["id"], line_count, clas, classDescription))
                                  
    # column 6 FN
    oneRow['FN'] = []
    for clas in row["classScoreCalculations"]:
        if clas['internalClassIndex'] in row["internalLabels"] and clas['internalClassIndex'] not in row["internalPrediction"]:
            oneRow['FN'].append(writeClass(row["id"], line_count, clas, classDescription))

    # column 7 TN
    oneRow['TN'] = []
    for clas in row["classScoreCalculations"]:
        if clas['internalClassIndex'] not in row["internalLabels"] and clas['internalClassIndex'] not in row["internalPrediction"]:
            oneRow['TN'].append(writeClass(row["id"], line_count, clas, classDescription))



def includesLabel(label, labels):
    for lb in labels:
        for key in lb:
            if label == lb[key]:
                return 1

    return 0


def createTable(data, fields,  classDescription):
    line_count = 0
    output = []
    for row in data:
        oneRow = {}
        line_count += 1
        r = []

        # column 1 IDs
        idlabels = {}
        idlabels['id'] = row['id'] 
        idlabels['internalId'] = row["internalId"]
        idlabels['internalLabels'] = row['internalLabels']
        idlabels['feedbackSelect'] = 'none'
        idlabels['feedbackText'] = ''

        # internal labels
        internalLabels = []
        releLabels = []
        for i in range(0, len(row["labels"])):
            label = {}
            label[row["internalLabels"][i]] = row["labels"][i]
            internalLabels.append(label)
            releLabels.append(row["labels"][i])
        idlabels['internalLabels'] = internalLabels
        # predictions
        predictions = []
        pres = []
        for i in range(0, len(row["prediction"])):
            label = {}
            label[row["internalPrediction"][i]] = row["prediction"][i]
            predictions.append(label)
            pres.append(row["prediction"][i])
        # if (len(predictions) == 0):
        #     label = {}
        #     label["-1"] = "N/A"
        #     predictions.append(label)
        #     pres.append("N/A")

        idlabels['predictions'] = predictions

        intersections = set(releLabels) & set(pres)
        unions = set(releLabels + pres)
        if len(unions) == 0:
            idlabels['overlap'] = "N/A"
        else:
            idlabels['overlap'] = "{0:.2f}".format(float(len(intersections)) / len(unions))
        if len(releLabels) == 0:
            idlabels['recall'] = "N/A"
        else:
            idlabels['recall'] = "{0:.2f}".format(float(len(intersections)) / len(releLabels))
        if len(pres) == 0:
            idlabels['precision'] = "N/A"
        else:
            idlabels['precision'] = "{0:.2f}".format(float(len(intersections)) / len(pres))

        oneRow['probForPredictedLabels'] = row['probForPredictedLabels']
        
        # column 2 predicted Ranking
        #predictedRanking = []
        #if row.has_key('predictedRanking'):
            #length = len(row["predictedRanking"])
        #else:
            #length = 0
        #for i in range(0, length):
            #predictedRanking.append(row["predictedRanking"][i])
        oneRow['predictedRanking'] = []

        for label in row["predictedRanking"]:
            if label["classIndex"] in row['internalLabels'] and label["classIndex"] in row['internalPrediction']:
                label["type"] = "TP"
            elif label["classIndex"] not in row['internalLabels'] and label["classIndex"] in row['internalPrediction']:
                label["type"] = "FP"
            elif label["classIndex"] in row['internalLabels'] and label["classIndex"] not in row['internalPrediction']:
                label["type"] = "FN"
            else:
                label["type"] = ""
            r.append(includesLabel(label["className"], internalLabels))
            oneRow['predictedRanking'].append(label)


        
        for labels in row["predictedLabelSetRanking"]:
            labels["types"] = []
            for index in labels["internalLabels"]:
                if index in row['internalLabels'] and index in row['internalPrediction']:
                    labels["types"].append("TP")
                elif index not in row['internalLabels'] and index in row['internalPrediction']:
                    labels["types"].append("FP")
                elif index in row['internalLabels'] and index not in row['internalPrediction']:
                    labels["types"].append("FN")
                else:
                    labels["types"].append("")
        oneRow["predictedLabelSetRanking"] = row["predictedLabelSetRanking"]

        prec = []
        sumOfR = float(0)
        sumOfPrec = 0
        last = 0
        for i in range(len(r)):
            if r[i] == 1:
                sumOfR += r[i]
                prec = sumOfR / (i + 1)
                sumOfPrec += prec
                last = i + 1

        if len(releLabels) == 0:
            idlabels['ap'] = "N/A"
        else:
            idlabels['ap'] = "{0:.2f}".format(float(sumOfPrec) / len(releLabels))
        if sumOfR < len(releLabels):
            idlabels['rankoffullrecall'] = "N/A"
        else:
            idlabels['rankoffullrecall'] = last
        oneRow['idlabels'] = idlabels
        
        # column 3 text
        res = es.get(index=esIndex, doc_type="document", id=row["id"])
        keys = fields
        oneRow["text"] = {}
        oneRow["others"] = {}
        for key in res["_source"]:
            if key in keys and isinstance(res["_source"][key], basestring):
                oneRow["text"][key] = res["_source"][key].encode('utf-8').replace("<", "&lt").replace(">", "&gt")
            else:
                oneRow["others"][key] = res["_source"][key]
        
        # column 4 - 7 TP FP FN TN columns
        ### pass classDescription
        createTFPNColumns(row, line_count, oneRow, classDescription)
        
        # finish row
        output.append(oneRow)

        if(line_count % 100 == 0):
            print "Current parsing ID: ", line_count


    return output

def createNewJsonForTopFeatures(inputData):
    outputData = []
    outputData.append([]) #classes
    outputData.append([]) #details
    indexes = {}

    for clas in inputData:
        feature = []
        feature.append(clas["classIndex"]) #0
        feature.append(clas["className"]) #1
        fds = []
        for fd in clas["featureDistributions"]:
            distribution = []
            if fd["feature"].has_key("ngram"):
                name = fd["feature"]["ngram"]
            else:
                name = fd["feature"]["name"]
            distribution.append(name) #0
            distribution.append([]) #occurrence
            distribution.append(fd["totalCount"])
            for occu in fd["occurrence"]:
                occurrence = []
                res = occu.rsplit(":", 1)
                className = res[0]
                r = res[1].split("/")

                if not indexes.has_key(className):
                    c = []
                    c.append(className) #0
                    c.append(r[1])  #num
                    outputData[0].append(c)
                    indexes[className] = len(outputData[0]) - 1
                    
                    
                occurrence.append(indexes[className])  # classindex
                occurrence.append(r[0]) #occu
                distribution[1].append(occurrence)
            fds.append(distribution)
        feature.append(fds)
        outputData[1].append(feature)

    return outputData


def parse(input_json_file, outputFileName, fields,  classDescription):
    # read input

    inputJson = open(input_json_file, "r")
    inputData = json.load(inputJson)
    print "Json:" + input_json_file + " load successfully.\nStart Parsing..."

    outputData = createTable(inputData, fields,  classDescription)
    outputJson = json.dumps(outputData)

    output = pre_data + outputJson + post_data

    outputFile = open(outputFileName, "w")
    outputFile.write(output)
    inputJson.close()
    outputFile.close()

def createTopFeatureHTML(input_json_file, outputFileName):
    output = ""
    try:
        inputJson = open(input_json_file, "r")
        inputData = json.load(inputJson)

        outputData = createNewJsonForTopFeatures(inputData)
        outputJson = json.dumps(outputData)

        output = pre_tf_data + outputJson + post_data
        inputJson.close()
    except IOError:
        print "Json: " + input_json_file + " no such file and skip it."
        output = pre_tf_data + post_data

    outputFile = open(outputFileName, "w")
    outputFile.write(output)
    outputFile.close()

def createIndPerHTML(inputIndPer, inputAllPer, outputPath):
    inputData = None
    if os.path.isfile(inputIndPer):
        inputJson = open(inputIndPer, "r")
        inputData = json.load(inputJson)

        inputAllJson = open(inputAllPer, "r")
        inputAllData = json.load(inputAllJson)


        outputJson = json.dumps(inputData)
        outputAllJson = json.dumps(inputAllData)

        output = pre_ind_data_part1 +"[" + outputAllJson + "]" + pre_ind_data_part2 + outputJson + post_data

        outputFile = open(outputPath, "w")
        outputFile.write(output)
        inputJson.close()
        inputAllJson.close()
        outputFile.close()



## BW: remove performance json from MetaData
# def createMetaDataHTML(inputData, inputModel, inputConfig, inputPerformance, outputFileName):
def createMetaDataHTML(inputData, inputModel, inputConfig, outputFileName):
    outputData = {}
    inputD = None
    inputM = None
    inputC = None
    # inputP = None
    if os.path.isfile(inputData):
        f1 = open(inputData, "r")
        inputD = json.load(f1)
    if os.path.isfile(inputModel):
        f2 = open(inputModel, "r")
        inputM = json.load(f2)
    if os.path.isfile(inputConfig):
        f3 = open(inputConfig, "r")
        inputC = json.load(f3)
    # if os.path.isfile(inputPerformance):
    #     f4 = open(inputPerformance, "r")
    #     inputP = json.load(f4)

    outputData["data"] = inputD
    outputData["model"] = inputM
    outputData["config"] = inputC
    # outputData["performance"] = inputP

    outputJson = json.dumps(outputData)

    output = pre_md_data + outputJson + post_data

    outputFile = open(outputFileName, "w")
    outputFile.write(output)

    if inputD != None:
        f1.close()
    if inputM != None:
        f2.close()
    if inputC != None:
        f3.close()
    # if inputP != None:
    #     f4.close()
    outputFile.close()


def parseAll(inputPath, directoryName, fileName, fields,  classFile):
    outputFileName = "viewer"

    indPerformanceName = "individual_performance"
    topName = "top_features"
    configName = "data_config"
    dataName = "data_info"
    modelName = "model_config"
    performanceName = "performance"
    inputTop = directoryName + topName + ".json"
    outputPath = directoryName + topName + ".html"
    createTopFeatureHTML(inputTop, outputPath)

    ## create individual performance html
    inputIndPer = directoryName + indPerformanceName + ".json"
    inputAllPer = directoryName + performanceName + ".json"
    outputPath = directoryName + indPerformanceName + ".html"
    createIndPerHTML(inputIndPer, inputAllPer, outputPath)

    inputData = directoryName + dataName + ".json"
    inputModel = directoryName + modelName + ".json"
    inputConfig = directoryName + configName + ".json"
    # inputPerformance = directoryName + performanceName + ".json"
    outputPath = directoryName + "metadata.html"
    # createMetaDataHTML(inputData, inputModel, inputConfig, inputPerformance, outputPath)
    createMetaDataHTML(inputData, inputModel, inputConfig, outputPath)

    ## reading class file if existing.
    classDescription = {}
    if classFile == "":
        print "no class description as input"
        # no class description as input.
    else:
        try:
            openClassFile = open(classFile, "r")
        except IOError:
            print "cannot open class file: ", classFile
        lines = openClassFile.readlines()
        for line in lines:
            line = line.translate(None, "\r\n")
            line = line.decode("utf-8")
            lineInfo = line.split("\t")
            className = lineInfo[0]
            classDesc = lineInfo[1]
            classDescription[className] = classDesc 
        openClassFile.close()


    ## skipJsonFiles are not default files: reports.json
    skipJsonFiles = [configName+".json", dataName + ".json", modelName + ".json", topName + ".json",
        performanceName + ".json", indPerformanceName + ".json"]
    if os.path.isfile(inputPath):
        parse(inputPath, directoryName + outputFileName + "_" + fileName[:-5] + ".html", fields,  classDescription)
    else:
        if not inputPath.endswith('/'):
            directoryName += '/'
        if not os.path.exists(directoryName):
            os.makedirs(directoryName)
        for f in listdir(inputPath):
            absf = join(inputPath, f)
            if not (isfile(absf) and f.endswith(".json")):
                continue
            elif f in skipJsonFiles:
                continue
            else:
                outputPath = directoryName + outputFileName + "_" + f[:-5] + ".html"
                parse(absf, outputPath, fields,  classDescription)

#constant Strings
pre_md_data = ''' <html>
    <head>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
        <script src="./js/jquery.min.js"></script>
    </head>
    <body><br>
        <table id="mytable" border=1  align="center" style="width:100%">
            <caption> Report </caption>
                <thead><tr>
                    <td align="center" width="30%"><b>data&nbspinfo</b></td>
                    <td align="center" width="30%"><b>data&nbspconfig</b></td>
                    <td align="center" width="30%"><b>model&nbspconfig</b></td>
                </tr></thead>
            <tbody id="data-table"></tbody>
        </table>
        <script>
            function dataFromJson() {
                return JSON.parse($('#raw-data').html())
            }

            function sortByViewOptions(data, displayOptions) {
                return data
            }

            function displayData(data) {
                str = ''

                for (key in data) {
                    str += '<br>' + key + ": " + data[key]
                }

                return str
            }

            function displayModel(model) {
                str = ''

                for (key in model) {
                    str += '<br>' + key + ": " + model[key]
                }

                return str
            }

            function displayConfig(config) {
                str = ''

                for (key in config) {
                    str += '<br>' + key + ": " + config[key]
                }

                return str
            }

            function render(data, displayOptions) {
                var $body = $('#data-table')
                $body.empty()
                var html = ''

                html += '<tr>' +
                    "<td style='vertical-align:top;text-align:left;'>" + 
                    displayData(data["data"]) + "</td>" +
                    "<td style='vertical-align:top;text-align:left;' rowspan='3'>" + 
                    displayConfig(data["config"]) + "</td>" +
                    "<td style='vertical-align:top;text-align:left;' rowspan='3'>" +
                    displayModel(data["model"]) + "</td>" +
                    '</tr>'

                $body.append(html)
            }

            function refresh() {
                var displayOptions = ""
                render(sortByViewOptions(dataFromJson(), displayOptions), displayOptions)
            }

            $(document).ready(function () {
                refresh()
            })
        </script>
        <style>
            #mytable{
                border: 1px solid black;
                border-collapse: collapse;
                word-wrap: break-word;
                table-layout:fixed;
            }
        </style>
    <script id="raw-data" type="application/json">
'''

pre_tf_data = ''' <html>
    <head>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
        <script src="./js/jquery.min.js"></script>
    </head>
    <body><br>
        <table id="mytable" border=1  align="center" style="width:100%">
            <caption> Report </caption>
                <thead><tr>
                    <td align="center" width="50%"><b>topFeatures</b></td>
                    <td align="center" width="50%"><b>featureDistributions</b></td>
                </tr></thead>
            <tbody id="data-table"></tbody>
        </table>

        <script>
            function dataFromJson() {
                return JSON.parse($('#raw-data').html())
            }

            function render(data, displayOptions) {
                var $body = $('#data-table')
                $body.empty()
                var html = ''
                classes = data[0]
                details = data[1]
                details.forEach(function (row, i) {
                    var labels = ''
                    html += '<tr>' +
                        "<td style='vertical-align:top;text-align:left;'>" + 
                        "<br>" + row[0] + "." + row[1] +
                        "<br>" + displayTopFeatures(row[2], i) + '</td>' +
                        displayFeatureDistributions(i, details.length) +
                        + '</tr>'
                })
                $body.append(html)
            }

            function displayOccurrence(data, i, j) {
                str = ''
                classes = data[0]
                occurrence = data[1][i][2][j][1]
                totalCount = data[1][i][2][j][2]

                str += "<br><br>totalCount: " + totalCount + "<br>"

                for (k = 0; k < occurrence.length; k++) {
                    occu = occurrence[k]
                    str += classes[occu[0]][0] + ":" + occu[1] + "/" 
                        + classes[occu[0]][1] + "<br>"
                }

                return str
            }

            function showDistribution(i, j) {
                var table = document.getElementById("mytable");
                var rows = table.getElementsByTagName('tr');
                var cols = rows[1].children;
                var cell = cols[1];

                data = dataFromJson()

                str = ""
                str += displayOccurrence(data, i, j)

                cell.innerHTML = str
            }

            function serialize(a, cb) {
                var str = ''
                a.forEach(function (obj, i) {
                    if (cb) {
                        str += cb(obj, i)
                    } else {
                        str += obj
                    }
                })
                return str
            }

            function sortByViewOptions(data, displayOptions) {
                return data
            }


            function displayFeature(distribution, i, j) {
                str = ''
                style = "style='color:#0000FF; margin:0px; padding:0px;' onclick='showDistribution(" + i + ", " + j + ")'"
                str += '<button ' + style + '>' + distribution[0] + '</button>'

                return str
            }

            function displayTopFeatures(fds, i) {
                str = ''
                str += serialize(fds, function(distribution, j) {
                            str = ""
                            if (j != 0) {
                                str += ","
                            }
                            str += displayFeature(distribution, i, j)
                            return str
                        })

                return str
            }

            function displayFeatureDistributions(index, len) {
                str = ''
                if (index == 0) {
                    str += "<td style='vertical-align:top;text-align:left;' rowspan='" + len + "'></td>"
                }
                return str
            }

            function render(data, displayOptions) {
                var $body = $('#data-table')
                $body.empty()
                var html = ''
                classes = data[0]
                details = data[1]
                details.forEach(function (row, i) {
                    var labels = ''

                    html += '<tr>' +
                        "<td style='vertical-align:top;text-align:left;'>" + 
                        "<br>" + row[0] + "." + row[1] +
                        "<br>" + displayTopFeatures(row[2], i) + '</td>' +
                        displayFeatureDistributions(i, details.length) +
                        + '</tr>'


                })

                $body.append(html)
            }

            function refresh() {
            //console.log(dataFromJson()[0])

                var displayOptions = ""
                render(sortByViewOptions(dataFromJson(), displayOptions), displayOptions)
            }            

            $(document).ready(function () {
                refresh()
            })
        </script>

        <style>
            #mytable{
                border: 1px solid black;
                border-collapse: collapse;
                word-wrap: break-word;
                table-layout:fixed;
            }
        </style>

    <script id="raw-data" type="application/json">
'''

# Constant Strings
pre_data = '''<html>
    <head>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
        <!-- <script src="./js/jquery.min.js"></script> -->
    </head>
    <body><br>
        <table id='optionTable'  style='width:55%'>
            <tr><th><br> Data Viewer Options: 
            </th></tr>
            <tr><td><br>
                <p style="text-indent: 1em;">Max number of rules per class:&nbsp;&nbsp;
                <input id="ruleNum" type="number" name="ruleNum" value="12" min="1" style='width:3em' onchange="refresh()"></p>
            </td></tr>
            <tr><td>
                <p style="text-indent: 1em;">Number of labels in predictedRanking:&nbsp;&nbsp;
                <select id="numOfLabels" onchange="refresh()">
                    <option value=5>TOP 5
                    <option value=10 >TOP 10
                    <option value=15 >TOP 15
                    <option value=-1 >ALL
                    <option value=-2 selected>Show Each Until Only TN
                </select>
            </td></tr>
            <tr><td>
                <p style="text-indent: 1em;">Select the fields to display
                <input id="TP" type="checkbox" name="TP" value="TP" checked>TP
                <input id="FP" type="checkbox" name="FP" value="FP" checked>FP
                <input id="FN" type="checkbox" name="FN" value="FN" checked>FN
                <input id="TN" type="checkbox" name="TN" value="TN">TN</p>
            </td></tr>
            <tr><td>
                <p style="text-indent: 1em;">Sort Docs:
                <input id="Confide" type="radio" name="sd" value="confide">Confide
                <input id="Mistake" type="radio" name="sd" value="mistake" checked>Mistake
                <input id="test" type="radio" name="sd" value="test">By Id
            </td></tr>
            <tr><td>
                <p style="text-indent: 1em;">Sort Rules:
                <input id="abs" type="radio" name="sr" value="abs">Abs Descending
                <input id="ascending" type="radio" name="sr" value="ascending">Ascending
                <input id="descending" type="radio" name="sr" value="descending">Descending
                <input id="anti" type="radio" name="sr" value="anti" checked>Anticorrelation
            </td></tr>
            <tr><td>
                <p style="text-indent: 1em;">Rule display:
                <input id="details" type="checkbox" name="details" value="details">Details
            </td></tr>
            <tr><td>
                <p style="text-indent: 1em;">
                <a href="top_features.html" target="_blank">Top Features</a>
                <a href="metadata.html" target="_blank">Metadata</a>
                <a href="individual_performance.html" target="_blank">Performance</a>
            </td></tr>
            <tr><td>
                <center><button id="createFile">Create New HTML</button> 
                <a download="new.html" id="downloadlink" style="display: none">Download</a></center><br>
            </td></tr>
        </table><br><br>

        <p>Feedbacks:</p>
        <table id="feedbackTable" border=1>
            <thead><tr>
                <td align="center"><b>failure</b></td>
                <td align="center"><b>incomplete</b></td>
                <td align="center"><b>bad_rule</b></td>
                <td align="center"><b>bad_matching</b></td>
            </tr></thead>
            <tbody id="data-feedbackTable"></tbody>
        </table>

        <p>Rules and classes:</p>
        <p>(Press any rules would highlight matched keywords in the text if they exist)</p>

        <table id="mytable" border=1  align="center" style="width:100%">
            <caption> Report </caption>
                <thead><tr>
                    <td align="center" width="20%"><b>id & labels</b></td>
                    <td align="center" width="20%"><b>Text</b></td>
                    <td align="center" width="20%"><b>TP</b></td>
                    <td align="center" width="20%"><b>FP</b></td>
                    <td align="center" width="20%"><b>FN</b></td>
                    <td align="center" width="20%"><b>TN</b></td>
                </tr></thead>
            <tbody id="data-table"></tbody>
        </table>

        <script>
            function changeFeedback(row) {
                document.getElementById('feedback' + row).style.display = "block";

                refreshTable(viewOptions())
            }

            function download() {
                feedbacks = []

                var table = document.getElementById("mytable");
                var rows = table.getElementsByTagName("tr");
                for (var row = 0; row < rows.length; row++) {
                    var feedback = {}
                    myselect = document.getElementById('sel' + row)
                    if (myselect != null && (option = myselect.options[myselect.selectedIndex].value) != 'none') {
                        key = document.getElementsByTagName('pre')[row].firstChild.data.replace (/  +$/, '')
                        feedback[key] = {'option:': option, 'text': document.getElementById('feedback' + row).value}
                        feedbacks.push(feedback)
                    }
                }
                var text = JSON.stringify(feedbacks); 

                url = window.location.href
                var pom = document.createElement('a');
                pom.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
                pom.setAttribute('download', 'Viewer_with_Feedbacks.html');

                pom.style.display = 'none';
                document.body.appendChild(pom);

                pom.click();

                document.body.removeChild(pom);
            }

            function convertHighlightsIntoPositions(highlights) {
                var colors = ["red", "Magenta", "lime", "blue", "GreenYellow", "LightPink", 
                        "orange", "yellow", "LightSeaGreen", "Orchid"];
                positions = {}

                for (var i = 0; i < highlights.length; i++) {
                    color = colors[i % colors.length]
                    poses = highlights[i]
                    for (var j = 0; j < poses.length; j++) {
                        positions[poses[j][0]] = {'end': poses[j][1], 'color': color}
                    }
                }

                return positions
            }

            function convertArrayToString(arr) {
                str = ""

                for (j = 0; j < arr.length; j++) {
                    a = arr[j]
                    str += a.toString()
                    if (j != arr.length - 1) {
                        str += " "
                    }
                }
                return str
            }

            function convertStringToArray(str) {
                arr = []
                if (str == "") {
                    return []
                }

                splits = str.split(" ")
                for (i = 0; i < splits.length; i++) {
                    a = []
                    pairs = splits[i].split(",")
                    for (j = 0; j < pairs.length; j += 2) {
                        a.push([pairs[j], pairs[j + 1]])
                    }
                    arr.push(a)
                }

                return arr
            }

            function areSamePoses(pos1, pos2) {
                if (pos1.length != pos2.length) {
                    return false
                }

                for (k = 0; k < pos1.length; k++) {
                    if (pos1[k][0] != pos2[k][0] || pos1[k][1] != pos2[k][1]) {
                        return false
                    }
                }

                return true
            }

            function indexOfHighlights(poses, highlights) {
                for (i = 0; i < highlights.length; i++) {
                    highlight = highlights[i]
                    if (areSamePoses(poses, highlight)) {
                        return i
                    }
                }
                return -1
            }

            function highlightText(poses, rowNum, field) {
                var table = document.getElementById("mytable");
                var rows = table.getElementsByTagName('tr');
                var cols = rows[rowNum].children;
                var cell = cols[1];
                var tagName = "#highlights" + (rowNum - 1)
                var highlights = $(tagName).data('data');

                if (rowNum.length == 0) {
                    return
                }

                newText = ""
                for (key in highlights) {
                    hs = highlights[key]
                    if (key == field) {
                        if ((index = indexOfHighlights(poses, hs)) != -1) {
                            hs.splice(index, 1)
                        }
                        else {
                            hs.push(poses)
                        }
                    }

                    positions = convertHighlightsIntoPositions(hs)
                    keys = Object.keys(positions)
                    var text = document.getElementById(key + (rowNum - 1)).innerHTML;
                    newText += key + ":<br>"
                    start = 0
                    for (var i = 0; i < keys.length; i++) {
                        index = keys[i]
                        end = index
                        newText += text.substring(start, end) + 
                            text.substring(end, positions[index]['end']).fontcolor(positions[index]['color'])
                        start = positions[index]['end']
                    }
                    newText += text.substring(start, text.length) + "<br>"
                }

                cell.innerHTML = newText
                $(tagName).data('data', highlights);
            }

            function createOption(value, isSelected) {
                if (isSelected) {
                    return "<option value=" + value + " selected>" + value + "</option>"
                } else {
                    return "<option value=" + value + ">" + value + "</option>"
                }
            }

            function createTextarea(i, display, text) {
                if (display) {
                    return "<textarea id=feedback" + i + " maxlength='512' class='mytext' cols='40' rows='5'>" +
                        text + "</textarea>"
                } else {
                    return "<textarea id=feedback" + i + " maxlength='512' class='mytext' cols='40' rows='5' style='display:none'>" +
                        text + "</textarea>" 
                }
            }

            function displayFeedback(i, sel, text) {
                return "<select onchange='changeFeedback(" + i + ")' id=sel" + i + " >" + 
                createOption('none', sel=='none') + 
                createOption('failure', sel=='failure') + 
                createOption('incomplete', sel=='incomplete') + 
                createOption('bad_rule', sel=='bad_rule') + 
                createOption('bad_matching', sel=='bad_matching') + 
                "</select>" + 
                createTextarea(i, sel!='none', text)
                
            }

            function createNewHTML() {
                //var textFile = null
                var create = document.getElementById('createFile')
                create.addEventListener('click', function () {
                    feedbacks = {}
                    startStr = "<script id=\\"raw-data\\" type=\\"application/json\\">"
                    endStr = "<\/script><\/body><\/html>"

                    var table = document.getElementById("mytable");
                    var rows = table.getElementsByTagName("tr");
                    for (var row = 0; row < rows.length; row++) {
                        myselect = document.getElementById('sel' + row)
                        if (myselect != null && (option = myselect.options[myselect.selectedIndex].value) != 'none') {
                            key = parseInt(document.getElementsByTagName('pre')[row].firstChild.data.replace (/  +$/, ''))
                            feedbacks[key] = {'option': option, 'text': document.getElementById('feedback' + row).value}
                        }
                    }

                    data = dataFromJson()
                    data.forEach(function(row) {
                        key = parseInt(row.idlabels.id)
                        if (feedbacks[key] != undefined) {
                            row.idlabels.feedbackSelect = feedbacks[key]['option']
                            row.idlabels.feedbackText = feedbacks[key]['text']
                        }
                        else {
                            row.idlabels.feedbackSelect = 'none'
                        }
                    })
                    dataJson = JSON.stringify(data);
                    text = document.documentElement.innerHTML
                    finalHTML = text.substring(0, text.indexOf(startStr)) + startStr + dataJson + endStr
                    var link = document.getElementById('downloadlink');
                    link.href = makeTextFile(finalHTML);
                    link.style.display = 'block';
                }, false);
            }

            function refresh() {
                var displayOptions = viewOptions()
                render(sortByViewOptions(dataFromJson(), displayOptions), displayOptions)
            }

            function generateFeedbackDataTable(data) {
                feedbacks = {'failure':0, 'incomplete':0, 'bad_rule':0, 'bad_matching':0}
                var $body = $('#data-feedbackTable')
                $body.empty()

                data.forEach(function (row, i) {
                    feedbacks[row.idlabels.feedbackSelect] += 1
                })

                var html = ''
                html += '<tr>' +
                    "<td style='vertical-align:top;text-align:left;'>" + feedbacks['failure'] + '</td>' +
                    "<td style='vertical-align:top;text-align:left;'>" + feedbacks['incomplete'] + '</td>' +
                    "<td style='vertical-align:top;text-align:left;'>" + feedbacks['bad_rule'] + '</td>' +
                    "<td style='vertical-align:top;text-align:left;'>" + feedbacks['bad_matching'] + '</td>' +
                    +'</tr>'

                $body.append(html)
            }

            function getLabelColor(type) {
                if (type == "TP") {
                    return "green"
                } 
                else if (type == "FN") {
                    return "green"
                }
                else if (type == "TN") {
                    return "black"
                }
                else if (type == "FP") {
                    return "red"
                }
                else {
                    return ""
                }
            }

            function includesLabel(clas, label) {
                for (i = 0; i < clas.length; i++) {
                    if (clas[i].name == label) {
                        return true
                    }
                }
                return false
            }  

            function displayPredictedRanking(row, displayOptions) {
                numOfLabels = 0
                predictedRanking = row.predictedRanking
                var split = false

                if (displayOptions.numOfLabels == -1) {
                    numOfLabels = predictedRanking.length
                } else if (displayOptions.numOfLabels == -2) {
                    for (i = 0; i < predictedRanking.length; i++) {
                        if (predictedRanking[i].type != "") {
                            numOfLabels = i + 1
                        }
                    }
                } else {
                    numOfLabels = displayOptions.numOfLabels
                }

                if (numOfLabels > 0) {
                    return serialize(predictedRanking.slice(0, numOfLabels), function (lb) {
                            var str = ''
                            text = lb.className + '(' + lb.prob.toFixed(2) + ')'
                            if (split == false && lb.prob.toFixed(2) < 0.5) {
                                split = true
                            }
                            if (lb.type == "TP" || lb.type == "FP") {
                                str += '<li><span style="background-color:lightGray">' + text.fontcolor(getLabelColor(lb.type)) + '</span></li>'
                            } else {
                                str += '<li style="list-style-type:none;">&nbsp&nbsp&nbsp' + text.fontcolor(getLabelColor(lb.type)) + '</li>'
                            }

                            return str
                    }) 
                } else {
                    return ""
                }
            }

            function labelsToString(labels) {
                var str = ''

                if (labels.length == 0) {
                    return str
                }

                str += labels[0]

                for (var i = 1; i < labels.length; i++) {
                    str += "|" + labels[i]
                }

                return str
            }

            function displayLabelSetRanking(row) {
                predictedLabelSetRanking = row.predictedLabelSetRanking
                str = ''

                for (var i = 0; i < predictedLabelSetRanking.length; i++) {
                    labels = predictedLabelSetRanking[i]
                    if (labels.labels.length == 0) {
                        temp = "EMPTY_SET".fontcolor("black")
                        temp += "(" + labels.probability.toFixed(2) +")"
                    } else {
                        temp = labels.labels[0].fontcolor(getLabelColor(labels.types[0]))
                        for (var j = 1; j < labels.labels.length; j++) {
                            temp += " | " + labels.labels[j].fontcolor(getLabelColor(labels.types[j]))
                        }

                        temp += '(' + labels.probability.toFixed(2)  + ')'
                    }

                    if (i == 0) {
                        temp = '<span style="background-color:lightGray">' + temp + '</span>'
                    }

                    temp = '<li>' + temp + '</li>'
                    str += temp
                }

                return str
            }

            function displayText(text) {
                keys = Object.keys(text)
                str = ''
                for (var i = 0; i < keys.length; i++) {
                    key = keys[i]
                    str += key + ":<br>" + text[key] + "<br>"
                }

                return str
            }

            function storeOrigText(text, index) {
                str = ""
                keys = Object.keys(text)
                for (var i = 0; i < keys.length; i++) {
                    str += "<pre id=" + keys[i] + index + " style='display:none'>" + text[keys[i]] + '</pre>'
                }

                return str
            }

            function displayOthers(others) {
                str = ''

                keys = Object.keys(others)
                for (i = 0; i < keys.length; i++) {
                    key = keys[i]
                    str += "<br><b>" + key + "</b>: " + others[key]
                }

                return str
            }

            function initialHighlights(data) {
                data.forEach(function (row, i) {
                    keys = Object.keys(row.text)
                    highlights = {}
                    for (j = 0; j < keys.length; j++) {
                        highlights[keys[j]] = []
                    }
                    $("#highlights" + i).data('data', highlights);
                })
            }

            function render(data, displayOptions) {
                generateFeedbackDataTable(data)

                var $body = $('#data-table')
                $body.empty()
                var html = ''
                data.forEach(function (row, i) {
                    var labels = ''

                    html += '<tr>' +
                        "<td style='vertical-align:top;text-align:left;' width='5%'>" + 
                        "<pre id='labelId" + i + "' style='display:none'>" + row.idlabels.id + '</pre>' +
                        "<input id='highlights" + i + "' style='display:none' value=''>" +
                        storeOrigText(row.text, i) +
                        "<b>ID:</b>&nbsp" + row.idlabels.id + 
                        displayOthers(row.others) + 
                        '<br><b>Labels</b>:' +  
                        serialize(row.idlabels.internalLabels, function (lb) {
                            var str = ''
                            for (var k in lb) {
                                str += '<li>' + lb[k] + '</li>'
                            }
                            return str
                         }) + 
                        '<br><b>Predictions</b>:' + 
                        serialize(row.idlabels.predictions, function (lb) {
                            var str = ''
                            for (var k in lb) {
                                str += '<li>' + lb[k] + '</li>'
                            }
                            return str
                        }) +
                        '<br><b>Label&nbspRanking</b>:' +
                        displayPredictedRanking(row, displayOptions) + 
                        "<br><b>AP:&nbsp</b>" + row.idlabels.ap +
                        "<br><b>RankOfFullRecall:&nbsp</b>" + row.idlabels.rankoffullrecall + "<br>" +
                        '<br><b>Label&nbspSet&nbspRanking</b>:' +
                        displayLabelSetRanking(row, displayOptions) + 
                        "<br><b>Overlap:&nbsp</b>" + row.idlabels.overlap +
                        "<br><b>Precision:&nbsp</b>" + row.idlabels.precision +
                        "<br><b>Recall:&nbsp</b>" + row.idlabels.recall +
                        '<br><br><br><b>Feedback</b>:' +
                        displayFeedback(i, row.idlabels.feedbackSelect, row.idlabels.feedbackText) +
                        '</td>' +
                        "<td style='vertical-align:top;text-align:left;'>" + 
                        displayText(row.text) +
                        '</td>' +
                        displayClass(row.TP, displayOptions, i) +
                        displayClass(row.FP, displayOptions, i) +
                        displayClass(row.FN, displayOptions, i) +
                        displayClass(row.TN, displayOptions, i) +
                        '</tr>'


                })

                $body.append(html)
                initialHighlights(data)
                refreshTable(displayOptions)
                createNewHTML()
            }


            function createRuleDetails(rule, displayOptions, rowNum) {
                str = ""
                score = ''
                if (rule.score >= 0) {
                    score += ": +" + Math.abs(rule.score).toExponential(2)
                } else {
                    score += ": -" + Math.abs(rule.score).toExponential(2)
                }
                str += '<li>' + serialize(rule['checks'], function (check) {
                            style = "style='color:#0000FF; margin:0px; padding:0px;' onclick='highlightText(" + check.highlights + ", " + 
                                (rowNum + 1) + ", \\"" + check.field + "\\")'"
                            //alert(Object.keys(check))
                            if ('ngram' in check) {
                                str = '<p ' + style + '>' + check.ngram + ' [' + check.value.toExponential(2) + check.relation + 
                                check.threshold.toExponential(2) +']'
                                if (displayOptions.details) {
                                    str += 'index=' + check.index + ' field=' + check.field + ' slop:' + check.slop
                                }
                            }
                            else {
                                str = '<p ' + style + '>' + check.name + ' [' + check.value.toExponential(2) + check.relation + 
                                check.threshold.toExponential(2) +']'
                                if (displayOptions.details) {
                                    str += 'index=' + check.index
                                }
                            }
                            return str
                        }) + '</li>'


                return str + score
            }

            // BW: highlight all rules by clicked label.
            // function highlightAll(rules, rowNum) {
            //    console.log(rules);
            //    console.log(rowNum);
            //    for (i=0; i<rules.length; i++) {
            //        rule = rules[i]
            //        for (j=0; j<rule['checks'].length; j++) {
            //            check = rule['checks'][j]
            //            highlightText(check.highlights, (rowNum+1), check.field)
            //        }
            //    }
            //}

            //function printTest(rules,rowNum){
            //    console.log(rules);
            //    console.log(rowNum)
            //}

            function displayClass(clas, displayOptions, rowNum) {
                str = ""
                str += "<td style='vertical-align:top;text-align:left;'>" +
                        serialize(clas, function (lb, i) {
                            str = ""
                            prior = ""
                            if (lb.prior != undefined) {
                                prior = '<li>prior: ' + lb.prior.toExponential(2) + '</li>'
                            }
                            if (i > 0) {
                                str += '<hr>'
                            }
                            // BW: add clickable style to label
                            style = "style='color:#00FF00; margin:0px; padding:0px;' onclick='highlightText(" + lb.allPos + "," + 
                                (rowNum + 1) + ", \\"" + lb.field + "\\")'"
                            str += '<p ' + style + '>' + lb.name + '</p>' + '<br><br>classProbability: ' + 
                            lb.classProbability.toExponential(2) + '<br><br>totalScore: ' + 
                            lb.totalScore.toExponential(2) + '<br><ul>' + 
                            prior +
                            serialize(lb.rules, function (rule, i) {
                                if (i >= displayOptions.ruleNum) {
                                    return ""
                                }  
                                return createRuleDetails(rule, displayOptions, rowNum)
                            })+ '</ul>'
                            return str
                        }) + '</td>'
                return str
            }

            function serialize(a, cb) {
                var str = ''

                if (a == undefined) {
                    return str
                }

                a.forEach(function (obj, i) {
                    if (cb) {
                        str += cb(obj, i)
                    } else {
                        str += obj
                    }
                })
                return str
            }

            function viewOptions() {
                var displayOptions = {}

                displayOptions.ruleNum = parseInt($('#ruleNum').val())
                displayOptions.fields = []
                if ($('#TP').prop('checked'))
                    displayOptions.fields.push('TP')
                if ($('#FP').prop('checked'))
                    displayOptions.fields.push('FP')
                if ($('#FN').prop('checked'))
                    displayOptions.fields.push('FN')
                if ($('#TN').prop('checked'))
                    displayOptions.fields.push('TN')

                displayOptions.sortDocs = $('input[name=sd]:checked').val()
                displayOptions.sortRules = $('input[name=sr]:checked').val()
                displayOptions.details = $('#details').prop('checked')
                numOfLabels = document.getElementById('numOfLabels')
                displayOptions.numOfLabels = numOfLabels.options[numOfLabels.selectedIndex].value
                return displayOptions
            }

            function sortByAbsScoreDescending(labels) {
                labels.forEach(function(lb) {
                    lb.rules = lb.rules.sort(function(a, b) {
                        return Math.abs(b.score) - Math.abs(a.score)
                    })
                })
            }

            function sortByScoreAscending(labels) {
                labels.forEach(function(lb) {
                    lb.rules = lb.rules.sort(function(a, b) {
                        return a.score - b.score
                    })
                })
            }

            function sortByScoreDescending(labels) {
                labels.forEach(function(lb) {
                    lb.rules = lb.rules.sort(function(a, b) {
                        return b.score - a.score
                    })
                })
            }

            function indexOfLabels(labels, label) {
                for (var i = 0; i < labels.length; i++) {
                    lb = labels[i]
                    for (var k in lb) {
                        if (lb[k] == label) {
                            return true
                        }
                    }
                }

                return false
            }

            function sortByAnti(labels, internalLabels) {
                labels.forEach(function(lb) {
                    if (indexOfLabels(internalLabels, lb.name) == true) {
                        lb.rules = lb.rules.sort(function(a, b) {
                            return a.score - b.score
                        })
                    } else {
                        lb.rules = lb.rules.sort(function(a, b) {
                            return b.score - a.score
                        })
                    }
                })
            }

            function sortByViewOptions(data, displayOptions) {
                if (displayOptions.sortRules === 'abs') {
                    data.forEach(function (row) {
                        sortByAbsScoreDescending(row.TP)
                        sortByAbsScoreDescending(row.FP)
                        sortByAbsScoreDescending(row.FN)
                        sortByAbsScoreDescending(row.TN)
                    })
                } else if (displayOptions.sortRules === 'ascending') {
                    data.forEach(function (row) {
                        sortByScoreAscending(row.TP)
                        sortByScoreAscending(row.FP)
                        sortByScoreAscending(row.FN)
                        sortByScoreAscending(row.TN)
                    })
                } else if (displayOptions.sortRules == 'descending') {
                    data.forEach(function (row) {
                        sortByScoreDescending(row.TP)
                        sortByScoreDescending(row.FP)
                        sortByScoreDescending(row.FN)
                        sortByScoreDescending(row.TN)
                    })
                } else if (displayOptions.sortRules == 'anti') {
                    data.forEach(function (row) {
                        sortByAnti(row.TP, row.idlabels.internalLabels)
                        sortByAnti(row.FP, row.idlabels.internalLabels)
                        sortByAnti(row.FN, row.idlabels.internalLabels)
                        sortByAnti(row.TN, row.idlabels.internalLabels)
                    })
                } else {
                    alert(displayOptions.sortRules)
                }

                // Sort data by displayOptions
                if (displayOptions.sortDocs == 'confide') {
                    data.sort(function (a, b) {
                        return a.idlabels.overlap - b.idlabels.overlap
                    })
                } else if (displayOptions.sortDocs == 'mistake') {
                    data.sort(function (a, b) {
                        return b.idlabels.overlap - a.idlabels.overlap
                    })
                } else {
                }
                return data
            }

            $(document).ready(function () {
                var displayOptions = viewOptions()
                render(sortByViewOptions(dataFromJson(), displayOptions), displayOptions) 

                $('#btn-submit').click(function () {
                    refresh()
                })
                $('#TP').click(function () {
                    refresh()
                })
                $('#FP').click(function () {
                    refresh()
                })
                $('#FN').click(function () {
                    refresh()
                })
                $('#TN').click(function () {
                    refresh()
                })
                $('#Confide').click(function () {
                    refresh()
                })
                $('#Mistake').click(function () {
                    refresh()
                })
                $('#abs').click(function () {
                    refresh()
                })
                $('#ascending').click(function () {
                    refresh()
                })
                $('#descending').click(function () {
                    refresh()
                })
                $('#anti').click(function () {
                    refresh()
                })
                $('#details').click(function () {
                    refresh()
                })
                $('#test').click(function () {
                    refresh()
                })
            })

            function makeTextFile(text) {
                var data = new Blob([text], {type: 'text/plain'});

                textFile = window.URL.createObjectURL(data);

                return textFile;
            }

            function refreshTable(displayOptions) {
                colums = ['TP', 'FP', 'FN', 'TN']
                feedbackColumns = ['failure', 'incomplete', 'bad_rule', 'bad_matching']
                selections = {'failure':0, 'incomplete':0, 'bad_rule':0, 'bad_matching':0}
                startIndex = 2

                var table = document.getElementById("mytable");
                var rows = table.getElementsByTagName("tr");
                for (var row = 0; row < rows.length; row++) {
                    var cols = rows[row].children;
                    var selId = 'sel' + row

                    sel = document.getElementById(selId)
                    if (sel != undefined) {
                        selections[sel.options[sel.selectedIndex].value] += 1
                    }

                    for (var i = 0; i < colums.length; i++) {
                        if (displayOptions.fields.indexOf(colums[i]) == -1) {
                            cols[startIndex + i].style.display = 'none';
                        } else {
                            cols[startIndex + i].style.display = 'table-cell';
                        }
                    }
                }
                
                var table = document.getElementById("data-feedbackTable")
                var row = table.getElementsByTagName("tr");
                cols = row[0].children
                for (i = 0; i < cols.length; i++) {
                    cols[i].innerHTML = selections[feedbackColumns[i]]
                }

            }

            function dataFromJson() {
                return JSON.parse($('#raw-data').html())
            }
        </script>

        <style>
            #feedbackTable{
                border: 1px solid black;
                border-collapse: collapse;
            }
            #mytable{
                border: 1px solid black;
                border-collapse: collapse;
                word-wrap: break-word;
                table-layout:fixed;
            }
            #optionTable {
                border : 1px solid black;
                border-collapse: collapse;
                float : center;
                width : 40%;
            }
            .ruleList {
                list-style-type: square;
            }
            .mytext {
                width: 100%;
                height:200px;
            }
            strong{
                font-weight:bold;
            }
        </style>
<script id="raw-data" type="application/json">
'''

post_data = '''
</script>
</body></html>
'''
pre_ind_data_part1 = '''
<html>
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
        <script type="text/javascript" class="init">
$(document).ready(function() {
    data = dataFromJson()
    allData = alldataFromJson()
    $('#overall').DataTable( {
        "aaData": allData,
        "iDisplayLength": 10,
        "aoColumns": [
            { "mDataProp": "instance subset accuracy" },
            { "mDataProp": "instance overlap" },
            { "mDataProp": "instance F1" },
            { "mDataProp": "instance Hamming loss" },
            { "mDataProp": "instance precision" },
            { "mDataProp": "instance recall"},
            { "mDataProp": "macro overlap" },
            { "mDataProp": "macro F1" },
            { "mDataProp": "macro Hamming loss" },
            { "mDataProp": "macro precision" },
            { "mDataProp": "macro recall" },
            { "mDataProp": "micro overlap" },
            { "mDataProp": "micro F1" },
            { "mDataProp": "micro Hamming loss" },
            { "mDataProp": "micro precision" },
            { "mDataProp": "micro recall" }
        ]
    } );
    $('#individual').DataTable( {
        "aaData": data,
        "iDisplayLength": 50,
        "aoColumns": [
            { "mDataProp": "label" },
            { "mDataProp": "TP" },
            { "mDataProp": "FP" },
            { "mDataProp": "TN" },
            { "mDataProp": "FN" },
            { "mDataProp": "accuracy" },
            { "mDataProp": "precision" },
            { "mDataProp": "recall" },
            { "mDataProp": "f1" }
        ]
    } );
} );
    </script>
    </head>
    <body><br>
        <h1>
            <span>Overall Performance</span>
        </h1>
        <table id="overall" class="display" width="100%">
            <thead>
        <tr>
            <th>instance subset accuracy</th>
            <th>instance overlap</th>
            <th>instance F1</th>
            <th>instance Hamming loss</th>
            <th>instance precision</th>
            <th>instance recall</th>
            <th>macro overlap</th>
            <th>macro F1</th>
            <th>macro Hamming loss</th>
            <th>macro precision</th>
            <th>macro recall</th>
            <th>micro overlap</th>
            <th>micro F1</th>
            <th>micro Hamming loss</th>
            <th>micro precision</th>
            <th>micro recall</th>
        </tr>
            </thead>
        </table>

        <h1>
            <span>Individual Label Performance</span>
        </h1>
        <table id="individual" class="display" width="100%">
            <thead>
        <tr>
            <th>label</th>
            <th>TP</th>
            <th>FP</th>
            <th>TN</th>
            <th>FN</th>
            <th>accuracy</th>
            <th>precision</th>
            <th>recall</th>
            <th>f1</th>
        </tr>
    </thead>
 
    <tfoot>
        <tr>
            <th>label</th>
            <th>TP</th>
            <th>FP</th>
            <th>TN</th>
            <th>FN</th>
            <th>accuracy</th>
            <th>precision</th>
            <th>recall</th>
            <th>f1</th>
        </tr>
    </tfoot>
        </table>
    <script>
            function dataFromJson() {
                return JSON.parse($('#raw-data').html())
            }
            function alldataFromJson() {
                return JSON.parse($('#all-data').html())
            }
    </script>

    <script id="all-data" type="application/json">
'''
pre_ind_data_part2 = '''
    </script>
    <script id="raw-data" type="application/json">
'''

es = Elasticsearch("localhost:9200", timeout=600, max_retries=10, revival_delay=0)
esIndex = "ohsumed_20000"
classNumber = 23

def main(argv):
    global esIndex
    global classNumber

    jsonFile = ""
    classFile = ""
    try:
        opts, args = getopt.getopt(argv,"hi:c:",["ifile=","cfile="])
    except getopt.GetoptError:
        print 'visualizer.py -i <inputfile> -c <classfile>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print 'visualizer.py -i <inputfile> -c <classfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            jsonFile = arg
        elif opt in ("-c", "--cfile"):
            classFile = arg

    # usage: myprog json_file
    splits = jsonFile.rsplit("/", 1)
    if len(splits) == 1:
        fileName = ""
        if os.path.isfile(jsonFile):
            directoryName = "./"
            fileName = splits[0]
        else:
            directoryName = splits[0] + "/"
    else:
        directoryName = jsonFile.rsplit("/", 1)[0] + '/'
        fileName = splits[1]
    # fileName = ""
    # if os.path.isfile(jsonFile):
    #     fileName = jsonFile

    # directoryName = ""
    # if not jsonFile.endswith("/"):
    #     directoryName = jsonFile + '/'
    # else:
    #     directoryName = jsonFile
    # print directoryName
    # raw_input()

    configName = "data_config.json"
    dataName = "data_info.json"
    modelName = "model_config.json"

    f1 = open(directoryName + configName, 'r')
    config1 = json.load(f1)
    f2 = open(directoryName + dataName, 'r')
    config2 = json.load(f2)
    f3 = open(directoryName + modelName, 'r')
    config3 = json.load(f3)

    esIndex = config1["index.indexName"]
    classNumber = config2["numClassesInModel"]
    fields = config1["index.ngramExtractionFields"]
    start = time.time()
    parseAll(jsonFile, directoryName, fileName, fields, classFile)
    end = time.time()
    print "parsing cost time ", end-start, " seconds"


if __name__ == '__main__':
    main(sys.argv[1:])




