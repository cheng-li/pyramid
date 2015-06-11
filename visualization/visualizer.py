#!/usr/bin/python
from elasticsearch import Elasticsearch
import json
import time
import sys


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
    text = res["hits"]["hits"][0]["_source"]["body"]
    textWithoutBr = text.replace("<br />", "")
    highlights = res["hits"]["hits"][0]["highlight"]["body"]
    for HL in highlights:

        cleanHL = HL.replace("<em>", "")
        cleanHL = cleanHL.replace("</em>", "")
        baseindex = text.find(cleanHL)

        # in case the highlight not found in body
        if baseindex == -1:
            continue
        curPos = len(text[0:baseindex].split())
        # returned highlight may cutoff words, so word position may minus 1
        if text[baseindex] != " " and curPos>0 and baseindex != 0 and text[baseindex-1] != " ":
            curPos = curPos - 1
        for word in HL.split():
            if word.find("<em>") != -1:
                positions.append(curPos)
            curPos += 1
    return positions


def writeRule(docId, line_count, num, rule):
    oneRule = {}

    oneRule['score'] = rule['score']
    oneRule['checks'] = []
    
    for check in rule["checks"]:
        #read pos from ElasticSearch
        checkOneRule = {}
        pos = [line_count]
        if check["feature value"] != 0.0 and check["feature"].has_key("ngram"):
            pos += getPositions(docId, check["feature"]["field"], check["feature"]["ngram"], 
                check["feature"]["slop"], check["feature"]["inOrder"])
        
        if not check["feature"].has_key("ngram"):
            checkOneRule["name"] = check["feature"]["name"]
            checkOneRule["index"] = check["feature"]["index"]
        else:
            checkOneRule["ngram"] = check["feature"]["ngram"]
            checkOneRule["index"] = check["feature"]["index"]
            checkOneRule["field"] = check["feature"]["field"]
            checkOneRule["slop"] = check["feature"]["slop"]

        checkOneRule['value'] = check["feature value"]
        checkOneRule['relation'] = check["relation"]
        checkOneRule['threshold'] = check["threshold"]
        checkOneRule['highlights'] = str(pos)

        oneRule['checks'].append(checkOneRule)
        
    return oneRule


def writeClass(docId, line_count, clas):
    oneClass = {}
    oneClass['id'] = clas["internalClassIndex"]
    oneClass['name'] = clas["className"]
    oneClass['classProbability'] = clas["classProbability"]
    oneClass['totalScore'] = clas["classScore"]

    # default rule number is 6
    oneClass['prior'] = clas["rules"][0]["score"]

    oneClass['rules'] = []
    for i in range(1, len(clas["rules"])):
        oneClass['rules'].append(writeRule(docId, line_count, i, clas["rules"][i]))

    return oneClass


def createTFPNColumns(row, line_count, oneRow):
    tmpDict = []
    # build set
    labelSet = set()
    for eachLabel in row["internalLabels"]:
        labelSet.add(eachLabel)
    predictionSet = set()
    for eachPredict in row["internalPrediction"]:
        predictionSet.add(eachPredict)  
        
    # column 4 TP
    oneRow['TP'] = []
    for eachLabel in row["internalLabels"]:
        if eachLabel in predictionSet:
            oneRow['TP'].append(writeClass(row["id"], line_count, row["classScoreCalculations"][eachLabel]))

    # column 5 FP
    oneRow['FP'] = []
    for eachPredict in row["internalPrediction"]:
        if eachPredict not in labelSet and eachPredict < len(row["classScoreCalculations"]):
            oneRow['FP'].append(writeClass(row["id"],line_count, row["classScoreCalculations"][eachPredict]))
                                  
    # column 6 FN
    oneRow['FN'] = []
    for eachLabel in row["internalLabels"]:
        if eachLabel not in predictionSet and eachLabel < len(row["classScoreCalculations"]):
            oneRow['FN'].append(writeClass(row["id"], line_count, row["classScoreCalculations"][eachLabel]))

    # column 7 TN
    oneRow['TN'] = []
    for i in range(0, classNumber):
        if (i not in labelSet) and (i not in predictionSet) and i < len(row["classScoreCalculations"]):
            oneRow['TN'].append(writeClass(row["id"],line_count, row["classScoreCalculations"][i]))


def createTable(data):
    line_count = 0
    output = []
    for row in data:
        oneRow = {}
        line_count += 1

        # column 1 IDs
        idlabels = {}
        idlabels['id'] = row['id']
        idlabels['internalId'] = row["internalId"]
        idlabels['internalLabels'] = row['internalLabels']
        idlabels['feedbackSelect'] = 'none'
        idlabels['feedbackText'] = ''

        # internal labels
        internalLabels = []
        for i in range(0, len(row["labels"])):
            label = {}
            label[row["internalLabels"][i]] = row["labels"][i]
            internalLabels.append(label)
        idlabels['internalLabels'] = internalLabels
        # predictions
        predictions = []
        for i in range(0, len(row["prediction"])):
            label = {}
            label[row["internalPrediction"][i]] = row["prediction"][i]
            predictions.append(label)
        idlabels['predictions'] = predictions
        oneRow['idlabels'] = idlabels
        
        # column 2 predicted Ranking
        predictedRanking = []
        if row.has_key('predictedRanking'):
            length = len(row["predictedRanking"])
        else:
            length = 0
        for i in range(0, length):
            predictedRanking.append(row["predictedRanking"][i])
        oneRow['probForPredictedLabels'] = row['probForPredictedLabels']
        oneRow['predictedRanking'] = predictedRanking
        
        # column 3 text
        res = es.get(index=esIndex, doc_type="document", id=row["id"])
        oneRow['text'] = res["_source"]["body"].encode('utf-8').replace("<", "&lt").replace(">", "&gt")
        
        # column 4 - 7 TP FP FN TN columns
        createTFPNColumns(row, line_count, oneRow)
        
        # finish row
        output.append(oneRow)
        # test break after first line
        if(line_count % 100 == 0):
            print "Current parsing ID: ", line_count
            break

    return output


def parse(input_json_file):
    # read input

    inputJson = open(input_json_file, "r")
    inputData = json.load(inputJson)
    print "Json load successfully.\nStart Parsing..."

    outputFile = open("Viewer.html", "w")

    outputData= createTable(inputData)
    outputJson = json.dumps(outputData)

    output = pre_data + outputJson + post_data


    outputFile.write(output)
    outputFile.close()


# Constant Strings
pre_data = '''<html>
    <head>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
        <script src="jquery.min.js"></script>
    </head>
    <body><br>
        <table id='optionTable'>
            <tr><th><br> Data Viewer Options: 
            </th></tr>
            <tr><td><br>
                <p style="text-indent: 1em;">Max number of rules per class:&nbsp;&nbsp;
                <input id="ruleNum" type="number" name="ruleNum" value="6" min="1" style='width:3em' onchange="refresh()"></p>
            </td></tr>
            <tr><td>
                <p style="text-indent: 1em;">Number of labels in predictedRanking:&nbsp;&nbsp;
                <select id="numOfLabels" onchange="refresh()">
                    <option value=5 selected>TOP 5
                    <option value=1 >TOP 10
                    <option value=15 >TOP 15
                    <option value=-1 >ALL
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
                <input id="Confide" type="radio" name="sd" value="confide" checked>Confide
                <input id="Mistake" type="radio" name="sd" value="mistake">Mistake
                <input id="test" type="radio" name="sd" value="test">By Id
            </td></tr>
            <tr><td>
                <p style="text-indent: 1em;">Sort Rules:
                <input id="abs" type="radio" name="sr" value="abs" checked>Abs Descending
                <input id="ascending" type="radio" name="sr" value="ascending">Ascending
                <input id="descending" type="radio" name="sr" value="descending">Descending
            </td></tr>
            <tr><td>
                <p style="text-indent: 1em;">Rule display:
                <input id="details" type="checkbox" name="details" value="details">Details
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
            <caption> XXX  data table</caption>
                <thead><tr>
                    <td align="center" width="5%"><b>id & labels</b></td>
                    <td align="center" width="5%"><b>predictedRanking</b></td>
                    <td align="center" width="10%"><b>Text</b></td>
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

            function highlightText(poses, rowNum) {
                var table = document.getElementById("mytable");
                var rows = table.getElementsByTagName('tr');
                var cols = rows[rowNum].children;
                var cell = cols[2];

                var text = cell.innerHTML;
                text = text.replace(/<font color=/g, "<font-color=");
                var words = text.split(" ");

                if(isNewHighLight(words, poses)) {
                    var colors = ["red", "Magenta", "lime", "blue", "GreenYellow", "LightPink", 
                        "orange", "yellow", "LightSeaGreen", "Orchid"];
                    var pickColor = pickNewColor(text, colors);

                    for(var i = 1; i < poses.length; i++) {
                        var pos = poses[i];
                        words[pos] = words[pos].replace(/(<([^>]+)>)/ig,"");
                        words[pos] = "<font color='" + pickColor + "'>"+words[pos]+"</font>";
                    }
                }
                else {
                    for(var i = 1; i < poses.length; i++) {
                        var pos = poses[i];
                        words[pos] = words[pos].replace(/(<([^>]+)>)/ig,"");
                    }
                }

                text = words.join(" ");
                cell.innerHTML = text.replace(/<font-color=/g, "<font color=");

            }

            function pickNewColor(text, colors) {
                for (var i=0; i<colors.length; i++) {
                    var fonttext = '<font-color="' + colors[i] + '">';
                    if(text.indexOf(fonttext) == -1) {
                        return colors[i];
                    }
                }
                return colors[0];
            }

            function isNewHighLight(words, rownum) {
                for(var i = 1; i < rownum.length; i++) {
                    if (words[rownum[i]].indexOf("<font-color=") == -1) {
                        return true;
                    }
                }
                return false;
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
                console.log(dataFromJson()[0])

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

            function render(data, displayOptions) {
                generateFeedbackDataTable(data)

                var $body = $('#data-table')
                $body.empty()
                var html = ''
                data.forEach(function (row, i) {
                    var labels = ''

                    html += '<tr>' +
                        "<td style='vertical-align:top;text-align:left;'>" + 
                        "<pre id='labelId" + i + "' style='display:none'>" + row.idlabels.id + '</pre>' +
                        "<br>ID:" + row.idlabels.id + " Iternal_ID: " + row.idlabels.internalId + 
                        '<br><br>Internal Labels:' +  
                        serialize(-1, row.idlabels.internalLabels, function (lb) {
                            var str = ''
                            for (var k in lb) {
                                str += '<br>' + k + ' : ' + lb[k] + '<br>'
                            }
                            return str
                         }) + 
                        '<br>Predictions:' +
                        serialize(-1, row.idlabels.predictions, function (lb) {
                            var str = ''
                            for (var k in lb) {
                                str += '<br>' + k + ' : ' + lb[k] + '<br>'
                            }
                            return str
                        }) + 
                        '<br>probForPredictedLabels: ' + row.probForPredictedLabels.toFixed(2) + 
                        '<br><br>Feedback:' +
                        displayFeedback(i, row.idlabels.feedbackSelect, row.idlabels.feedbackText) +
                        '</td>' +
                        "<td style='vertical-align:top;text-align:left;'> " + 
                        serialize(displayOptions.numOfLabels, row.predictedRanking, function (lb) {
                            var str = lb + '<br>'
                            return str
                        }) + '</td>' +
                        "<td style='vertical-align:top;text-align:left;'>" + row.text
                        + '</td>' +
                        displayClass(row.TP, displayOptions, i) +
                        displayClass(row.FP, displayOptions, i) +
                        displayClass(row.FN, displayOptions, i) +
                        displayClass(row.TN, displayOptions, i)
                        + '</tr>'


                })

                $body.append(html)
                refreshTable(displayOptions)
                createNewHTML()
            }


            function createRuleDetails(rule, displayOptions, rowNum) {
                str = ""
                score = ''
                if (rule.score >= 0) {
                    score += ": +" + Math.abs(rule.score.toFixed(2))
                } else {
                    score += ": -" + Math.abs(rule.score.toFixed(2))
                }
                str += '<li>' + serialize(-1, rule['checks'], function (check) {
                            style = "style='color:#0000FF; margin:0px; padding:0px;' onclick='highlightText(" + check.highlights + ", " + 
                                (rowNum + 1) + ")'"
                            // alert(Object.keys(check))
                            if ('ngram' in check) {
                                str = '<p ' + style + '>' + check.ngram + ' [' + check.value.toFixed(2) + check.relation + 
                                check.threshold.toFixed(2) +']'
                                if (displayOptions.details) {
                                    str += 'index=' + check.index + ' field=' + check.field + ' slop:' + check.slop
                                }
                            }
                            else {
                                str = '<p ' + style + '>' + check.name + ' [' + check.value.toFixed(2) + check.relation + 
                                check.threshold.toFixed(2) +']'
                                if (displayOptions.details) {
                                    str += 'index=' + check.index
                                }
                            }
                            return str
                        }) + '</li>'


                return str + score
            }


            function displayClass(clas, displayOptions, rowNum) {
                str = ""
                str += "<td style='vertical-align:top;text-align:left;'>" +
                        serialize(-1, clas, function (lb) {
                            return lb.id + ' : ' + lb.name + '<br><br>classProbability: ' + 
                            lb.classProbability.toFixed(2) + '<br><br>totalScore: ' + 
                            lb.totalScore.toFixed(2) + '<br><ul>' + '<li>prior: ' + 
                            lb.prior.toFixed(2) + '</li>' +
                            serialize(-1, lb.rules, function (rule, i) {
                                if (i >= displayOptions.ruleNum) {
                                    return ""
                                }  
                                return createRuleDetails(rule, displayOptions, rowNum)
                            })+ '</ul>'
                        }) + '</td>'
                return str
            }

            function serialize(num, a, cb) {
                var str = ''
                a.forEach(function (obj, i) {
                    if (num > 0 && i >= num) {
                        return str
                    }
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

            function sortByScoreAscending(a, b) {
                labels.forEach(function(lb) {
                    lb.rules = lb.rules.sort(function(a, b) {
                        return a.score - b.score
                    })
                })
            }

            function sortByScoreDescending(a, b) {
                labels.forEach(function(lb) {
                    lb.rules = lb.rules.sort(function(a, b) {
                        return b.score - a.score
                    })
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
                        sortByScoreDescending(row.TP)
                        sortByScoreDescending(row.FP)
                        sortByScoreDescending(row.FN)
                        sortByScoreDescending(row.TN)
                } else {
                    alert(displayOptions.sortRules)
                }

                // Sort data by displayOptions
                if (displayOptions.sortDocs == 'confide') {
                    data.sort(function (a, b) {
                        return a.probForPredictedLabels - b.probForPredictedLabels
                    })
                } else if (displayOptions.sortDocs == 'mistake') {
                    data.sort(function (a, b) {
                        return b.probForPredictedLabels - a.probForPredictedLabels
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

                var table = document.getElementById("mytable");
                var rows = table.getElementsByTagName("tr");
                for (var row = 0; row < rows.length; row++) {
                    var cols = rows[row].children;
                    var selId = 'sel' + row

                    for (var i = 0; i < colums.length; i++) {
                        if (displayOptions.fields.indexOf(colums[i]) == -1) {
                            cols[i + 3].style.display = 'none';
                        } else {
                            cols[i + 3].style.display = 'table-cell';
                        }
                    }
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
                width: 200px;
                height:200px;
            }
        </style>
<script id="raw-data" type="application/json">
'''
post_data = '''
</script>
</body></html>
'''


es = Elasticsearch("localhost:9200", timeout=600, max_retries=10, revival_delay=0)
esIndex = "ohsumed_20000"
classNumber = 23

def main():
    global esIndex
    global classNumber
    # usage: myprog json_file
    if len(sys.argv) >= 3:
        esIndex = sys.argv[1]
        classNumber = int(sys.argv[2])
        jsonFile = sys.argv[3]
    else:
        print "Usage: python Visualizor.py index NumberofClasses json_file"
        return
    start = time.time()
    parse(jsonFile)
    end = time.time()
    print "parsing cost time ", end-start, " seconds"


if __name__ == '__main__':
    main()




