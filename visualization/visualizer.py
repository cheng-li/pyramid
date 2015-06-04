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
    text = text.replace("<br />", "")
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
        if text[baseindex] != " " and curPos>0 and text[baseindex-1] != " ":
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
            pos += getPositions(docId, check["feature"]["field"], check["feature"]["ngram"], check["feature"]["slop"], check["feature"]["inOrder"])
        
        if check["feature"].has_key("ngram"):
            checkOneRule['ngram'] = check["feature"]["ngram"]
        else:
            checkOneRule['ngram'] = ""
            
        checkOneRule['index'] = check["feature"]["index"]
        checkOneRule['field'] = check["feature"]["field"]
        checkOneRule['slop'] = check["feature"]["slop"]
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
        for i in range(0, len(row["predictedRanking"])):
            predictedRanking.append(row["predictedRanking"][i])
        oneRow['probForPredictedLabels'] = row['probForPredictedLabels']
        oneRow['predictedRanking'] = predictedRanking
        
        # column 3 text
        res = es.get(index=esIndex, doc_type="document", id=row["id"])
        oneRow['text'] = res["_source"]["body"].encode('utf-8')
        
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
<body>
<br>
<table id='optionTable'>
  <tr><th><br> Data Viewer Options: </th></tr>
  <tr><td><br>
      <p style="text-indent: 1em;">Max number of rules per class:&nbsp;&nbsp;
    <input id="ruleNum" type="number" name="ruleNum" value="6" min="1" style='width:3em'>   <button id="btn-submit">Update</button></p>
  </td></tr><tr><td>
      <p style="text-indent: 1em;">Select the fields to display
    <input id="TP" type="checkbox" name="TP" value="TP" checked>TP
    <input id="FP" type="checkbox" name="FP" value="FP" checked>FP
    <input id="FN" type="checkbox" name="FN" value="FN" checked>FN
    <input id="TN" type="checkbox" name="TN" value="TN">TN</p>
  </td></tr><tr><td>
      <p style="text-indent: 1em;">Sort Docs:
    <input id="Confide" type="radio" name="sd" value="confide" checked>Confide
    <input id="Mistake" type="radio" name="sd" value="mistake">Mistake
    <input id="test" type="radio" name="sd" value="test">By Id
  </td></tr><tr><td>
      <p style="text-indent: 1em;">Sort Rules:
    <input id="abs" type="radio" name="sr" value="abs" checked>Abs Descending
    <input id="anticorrelation" type="radio" name="sr" value="anticorrelation">Ascending
    <input id="pos" type="radio" name="sr" value="pos">Descending
  </td></tr><tr><td>
      <p style="text-indent: 1em;">Rule display:
    <input id="details" type="checkbox" name="details" value="details">Details
  </td></tr><tr><td>
      <center><button id="create">Create file</button> <a download="new.html" id="downloadlink" style="display: none">Download</a></center>
      <br></td></tr>
</table>

<br><br>

<p>Rules and classes:</p>
<p>(Press any rules would highlight matched keywords in the text if they exist)</p>

<table id="mytable" border=1  align="center" style="width:100%">
  <caption> XXX  data table</caption>
  <thead>
    <tr>
      <td align="center" width="5%"><b>id & labels</b></td>
      <td align="center" width="5%"><b>predictedRanking</b></td>
      <td align="center" width="10%"><b>Text</b></td>
      <td align="center" width="20%"><b>TP</b></td>
      <td align="center" width="20%"><b>FP</b></td>
      <td align="center" width="20%"><b>FN</b></td>
      <td align="center" width="20%" style="display:none"><b>TN</b></td>
    </tr>
  </thead>

  <tbody id="data-table">
  </tbody>
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

    (function () {
        var _data = null

        function render(data, options) {
            var $body = $('#data-table')
            $body.empty()
            var html = ''
            data.forEach(function (row, i) {
                var labels = ''

                html += '<tr>' +
                    "<td style='vertical-align:top;text-align:left;' width='5%'>" + 
                    "<pre id='labelId" + i + "' style='display:none'>" + row.idlabels.id + '</pre>' +
                    "<br>ID:" + row.idlabels.id + " Iternal_ID: " + row.idlabels.internalId + 
                    '<br><br>Internal Labels:' +  
                    serialize(row.idlabels.internalLabels, function (lb) {
                        var str = ''
                        for (var k in lb) {
                            str += '<br>' + k + ' : ' + lb[k] + '<br>'
                        }
                        return str
                     }) + 
                    '<br>Predictions:' +
                    serialize(row.idlabels.predictions, function (lb) {
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
                    "<td style='vertical-align:top;text-align:left;' width='5%'> " + 
                    serialize(row.predictedRanking, function (lb) {
                        var str = lb + '<br>'
                        return str
                    }) + '</td>' +
                    "<td style='vertical-align:top;text-align:left;' width='10%'>" + row.text
                    + '</td>' +
                    displayClass(row.TP, options, i) +
                    displayClass(row.FP, options, i) +
                    displayClass(row.FN, options, i) +
                    displayClass(row.TN, options, i)


            })

            $body.append(html)
            refreshTable(data, options)
        }

        function createRuleDetails(rule, options, rowNum) {
            str = ""
            score = ''
            if (rule.score >= 0) {
                score += ": +" + Math.abs(rule.score.toFixed(2))
            } else {
                score += ": -" + Math.abs(rule.score.toFixed(2))
            }
            str += serialize(rule['checks'], function (check) {
                        style = "style='color:#0000FF' onclick='highlightText(" + check.highlights + ", " + 
                            (rowNum + 1) + ")'"
                        str = '<li ' + style + '>' + check.ngram + ' [' + check.value.toFixed(2) + check.relation + 
                        check.threshold.toFixed(2) +']'
                        if (options.details) {
                            str += 'index=' + check.index + ' field=' + check.field + ' slop:' + check.slop
                        }
                        str += '</li>'
                        return str
                    })
 

            return str + score
        }
        
        function displayClass(clas, options, rowNum) {
            str = ""
            str += "<td style='vertical-align:top;text-align:left;' width='12%'>" +
                    serialize(clas, function (lb) {
                        return lb.id + ' : ' + lb.name + '<br><br>classProbability: ' + 
                        lb.classProbability.toFixed(2) + '<br><br>totalScore: ' + 
                        lb.totalScore.toFixed(2) + '<br><ul>' + '<li>prior: ' + 
                        lb.prior.toFixed(2) + '</li>' +
                        serialize(lb.rules, function (rule, i) {
                            if (i >= options.ruleNum) {
                                return ""
                            }  
                            return createRuleDetails(rule, options, rowNum)
                        })+ '</ul>'
                    }) + '</td>'
            return str
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

        function viewOptions() {
            var options = {}

            options.ruleNum = parseInt($('#ruleNum').val())
            options.fields = []
            if ($('#TP').prop('checked'))
                options.fields.push('TP')
            if ($('#FP').prop('checked'))
                options.fields.push('FP')
            if ($('#FN').prop('checked'))
                options.fields.push('FN')
            if ($('#TN').prop('checked'))
                options.fields.push('TN')

            options.sortDocs = $('input[name=sd]:checked').val()
            options.sortRules = $('input[name=sr]:checked').val()
            options.details = $('#details').prop('checked')

            return options
        }

        function sortByViewOptions(data, options) {
            if (options.sortRules === 'abs') {
                data.forEach(function (row) {
                    row.TP.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                            return Math.abs(a.score) < Math.abs(b.score)
                        })
                    })
                    row.FP.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                            return Math.abs(a.score) < Math.abs(b.score)
                        })
                    })
                    row.FN.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                            return Math.abs(a.score) < Math.abs(b.score)
                        })
                    })
                    row.TN.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                            return Math.abs(a.score) < Math.abs(b.score)
                        })
                    })
                })
            } else if (options.sortRules === 'anticorrelation') {
                data.forEach(function (row) {
                    row.TP.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                           return a.score > b.score
                        })
                    })
                })
                data.forEach(function (row) {
                    row.FP.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                           return a.score > b.score
                        })
                    })
                })
                data.forEach(function (row) {
                    row.FN.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                           return a.score > b.score
                        })
                    })
                })
                data.forEach(function (row) {
                    row.TN.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                           return a.score > b.score
                        })
                    })
                })
            } else if (options.sortRules == 'pos') {
                data.forEach(function (row) {
                    row.TP.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                           return a.score < b.score
                        })
                    })
                })
                data.forEach(function (row) {
                    row.FP.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                           return a.score < b.score
                        })
                    })
                })
                data.forEach(function (row) {
                    row.FN.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                           return a.score < b.score
                        })
                    })
                })
                data.forEach(function (row) {
                    row.TN.forEach(function (lb) {
                        lb.rules = lb.rules.sort(function (a, b) {
                           return a.score < b.score
                        })
                    })
                })
            } else {
                alert(options.sortRules)
            }

            // Sort data by options
            if (options.sortDocs == 'confide') {
                data.sort(function (a, b) {
                    return a.probForPredictedLabels - b.probForPredictedLabels
                })
            } else if (options.sortDocs == 'mistake') {
                data.sort(function (a, b) {
                    return b.probForPredictedLabels - a.probForPredictedLabels
                })
            } else {
            }
            return data
        }

        function dataFromJson() {
            return JSON.parse($('#raw-data').html())
        }

        function refresh() {
            console.log(dataFromJson()[0])

            var options = viewOptions()
            render(sortByViewOptions(dataFromJson(), options), options)
        }

        $(document).ready(function () {
            var options = viewOptions()
            render(sortByViewOptions(dataFromJson(), options), options) 

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
            $('#anticorrelation').click(function () {
                refresh()
            })
            $('#pos').click(function () {
                refresh()
            })
            $('#details').click(function () {
                refresh()
            })
            $('#test').click(function () {
                refresh()
            })
        })

        function refreshTable(options) {

            var ruleNum = document.getElementById("ruleNum").value;

            if (ruleNum == '') {
                alert("Wrong Number input");
                return;
            }

            ruleNum = ruleNum * 2 + 1;

            var rulelists = document.getElementsByClassName("ruleList");

            for (var i=0; i<rulelists.length; i++) {
                var singleRuleLists = rulelists[i].getElementsByTagName("li");
                for (var j=0; j<singleRuleLists.length && j<ruleNum; j++) {
                    singleRuleLists[j].style.display = 'list-item';
                }
                for (var j=ruleNum; j<singleRuleLists.length; j++) {
                    singleRuleLists[j].style.display = 'none';
                }
            }

            var tp = document.getElementById("TP");
            var fp = document.getElementById("FP");
            var fn = document.getElementById("FN");
            var tn = document.getElementById("TN");

            var table = document.getElementById("mytable");
            var rows = table.getElementsByTagName("tr");
            for (var row = 0; row < rows.length; row++) {
                var cols = rows[row].children;
                var selId = 'sel' + row
                /*
                if (document.getElementById(selId) != null) {
                    document.getElementById(selId).value = '3'
                } else {
                    //alert(selId)
                }*/

                if ($.inArray("TP", options.fields) && cols[3].innerHTML != "" ||
                    $.inArray("FP", options.fields) && cols[4].innerHTML != "" ||
                    $.inArray("FN", options.fields) && cols[5].innerHTML != "" ||
                    $.inArray("TN", options.fields) && cols[6].innerHTML != "") {
                    cols[0].style.display = 'block';
                    cols[1].style.display = 'block';
                    cols[2].style.display = 'block';
                    cols[3].style.display = tp.checked ? 'block' : 'none';
                    cols[4].style.display = fp.checked ? 'block' : 'none';
                    cols[5].style.display = fn.checked ? 'block' : 'none';
                    cols[6].style.display = tn.checked ? 'block' : 'none';
                }
                else {
                    cols[0].style.display = 'none';
                    cols[1].style.display = 'none';
                    cols[2].style.display = 'none';
                    cols[3].style.display = 'none';
                    cols[4].style.display = 'none';
                    cols[5].style.display = 'none';
                    cols[6].style.display = 'none';
                }
            }

            var textFile = null
            makeTextFile = function (text) {
                var data = new Blob([text], {type: 'text/plain'});

                // If we are replacing a previously generated file we need to
                // manually revoke the object URL to avoid memory leaks.
                if (textFile !== null) {
                    window.URL.revokeObjectURL(textFile);
                }

                textFile = window.URL.createObjectURL(data);

                return textFile;
            }

            var create = document.getElementById('create')
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
}())
</script>

<style>
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




