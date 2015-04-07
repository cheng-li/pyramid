#!/usr/bin/python
from elasticsearch import Elasticsearch
import json
import time
import sys


# This program read in a json file data and output to a html file.
# The data is then shown in a big table with many options in front of the page.
# The javascript then adjust the style(block or none) of each cell to determine
# whether to show that in the table.


def writeHeader(output):
    output.write(header)


def getPositions(docId, field, keywords, slop, in_order):
    clauses = "["
    for keyword in keywords.split():
        clauses += "{'span_term':{'" + field + "':'" + keyword + "'}},"
    clauses = clauses[0:-1] + "]"
    res = es.search(index="ohsumed_20000",
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
                          "highlight":{"fields":{"body":{}}},
                          "size":10000})
    print res
    if len(res["hits"]["hits"]) <= 0:
        return []
    positions = []
    text = res["hits"]["hits"][0]["_source"]["body"]
    highlights = res["hits"]["hits"][0]["highlight"]["body"]
    for HL in highlights:
        cleanHL = HL.replace("<em>","")
        cleanHL = cleanHL.replace("</em>","")
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


def writeRule(docId, line_count, num, output, rule, show):
    if show:
        output.write("<li>")
    else:
        output.write("<li display:none>")
    output.write("rule%d: %.2f" % (num,rule["score"]))
    output.write("<ul>")
    for check in rule["checks"]:
        # read pos from ElasticSearch
        pos = [line_count]
        if check["feature value"] != 0.0:
            pos += getPositions(docId, check["feature"]["field"], check["feature"]["ngram"], check["feature"]["slop"], check["feature"]["inOrder"])
        style = "style='color:#0000FF' onclick='highlightText(%s)'" % str(pos)
        output.write("<li %s>index:%d <br>" % (style, check["feature"]["index"]))
        output.write("ngram:%s<br>field:%s<br>slop=%d<br>" % (check["feature"]["ngram"],check["feature"]["field"],check["feature"]["slop"]))
        output.write("%.2f %s %.2f<br></li>" % (check["feature value"],check["relation"],check["threshold"]))
    output.write("</ul>")
    output.write("</li>")
    return


def writeClass(docId, line_count, output, clas):
    output.write("<br>")
    output.write("%d : %s<br>" % (clas["internalClassIndex"], clas["className"]))
    output.write("classProbability: %.2f<br>" % clas["classProbability"] )
    output.write("TotalScore: %.2f<br>" % clas["classScore"])
    # default rule number is 5
    output.write("<ul>")
    output.write("<li>prior: %.2f</li>" % clas["rules"][0]["score"])
    for i in range(1, min(6,len(clas["rules"]))):
        writeRule(docId, line_count, i, output, clas["rules"][i], True)
    for i in range(6, len(clas["rules"])):
        writeRule(docId, line_count, i, output, clas["rules"][i], False)
    output.write("</ul>")


def writeTableTFPNColumns(row, line_count, output):
    # build set
    labelSet = set()
    for eachLabel in row["internalLabels"]:
        labelSet.add(eachLabel)
    predictionSet = set()
    for eachPredict in row["internalPrediction"]:
        predictionSet.add(eachPredict)
    # column 4 TP
    output.write("<td style='vertical-align:top;text-align:left;' width='12%'>")
    for eachLabel in row["internalLabels"]:
        if eachLabel in predictionSet:
            writeClass(row["id"],line_count, output, row["classScoreCalculations"][eachLabel])
    output.write("</td>")
    # column 5 FP
    output.write("<td style='vertical-align:top;text-align:left;' width='12%'>")
    for eachPredict in row["internalPrediction"]:
        if eachPredict not in labelSet:
            writeClass(row["id"],line_count, output, row["classScoreCalculations"][eachPredict])
    output.write("</td>")
    # column 6 FN
    output.write("<td style='vertical-align:top;text-align:left;' width='12%'>")
    for eachLabel in row["internalLabels"]:
        if eachLabel not in predictionSet:
            writeClass(row["id"],line_count, output, row["classScoreCalculations"][eachLabel])
    output.write("</td>")
    # column 7 TN
    output.write("<td style='vertical-align:top;text-align:left;display:none' width='12%'>")
    for i in range(0,23):
        if (i not in labelSet) and (i not in predictionSet):
            writeClass(row["id"],line_count, output, row["classScoreCalculations"][i])
    output.write("</td>")


def writeTable(output, data):
    line_count = 0
    for row in data:
        #if not (row["id"] == 16966 or row["id"] == 24362 or row["id"] == 19749):
        #    continue
        line_count += 1
        # row begin
        output.write("<tr>")
        # column 1 IDs
        output.write("<td style='vertical-align:top;text-align:left;' width='5%'>")
        output.write("<br>ID: %s " % (row["id"]))
        output.write("Internal_ID: %s <br>" % (row["internalId"]))
        for i in range(0, len(row["labels"])):
            output.write("<br>%d : %s<br>" % (row["internalLabels"][i], row["labels"][i]))
        output.write("<br></td>")
        # column 2 predictions
        output.write("<td style='vertical-align:top;text-align:left;' width='5%'>")
        output.write("<br><br><br>")
        for i in range(0, len(row["prediction"])):
            output.write("<br>%d : %s<br>" % (row["internalPrediction"][i], row["prediction"][i]))
        output.write("</td>")
        # column 3 text
        output.write("<td style='vertical-align:top;text-align:left;' width='15%'>")
        res = es.get(index="ohsumed_20000",
                     doc_type="document",
                     id=row["id"])
        output.write(res["_source"]["body"].encode('utf-8'))
        output.write("</td>")
        # column 4 - 7 TP FP FN TN columns
        writeTableTFPNColumns(row, line_count, output)
        # finish row
        output.write("</tr>")
        # test break after first line
        if(line_count % 1 == 0):
            break
            print "cur line: ", line_count


def writeBody(output, data):
    output.write("<body>")
    output.write(preBody)
    writeTable(output, data)
    output.write(postBody)
    output.write(scripts)
    output.write("</body>")


def parse(inputJsonFile):
    # read input
    input_json = open(inputJsonFile, "r")
    data = json.load(input_json)
    print "load finish"
    # define output
    output = open("Viewer.html", "w")
    # write html file
    output.write("<html>")
    writeHeader(output)
    writeBody(output, data)
    output.write("</html>")
    # Finish
    output.close()


# Constant Strings
header = '''
<head>
<style>
table, th, td {
border: 1px solid black;
border-collapse: collapse;
}
</style>
</head>
'''

preBody = '''
<p>This is a hospital's patients data table:</p>
<br>

<p>Number of rules per class:
<input type="text" name="ruleNum" value="6" size="4"></p>

<p>Please select the fields to display</p>
<input id="TP" type="checkbox" name="TP" value="TP" checked>TP
<input id="FP" type="checkbox" name="FP" value="FP" checked>FP
<input id="FN" type="checkbox" name="FN" value="FN" checked>FN
<input id="TN" type="checkbox" name="TN" value="TN" >TN
<button onclick="refreshTable()">Refresh</button>
<br><br>

<p>Rules and classes:</p>
<p>(Press any rules would highlight matched query in the text)</p>
<table id="mytable" border=1  align="center" style="width:100%">
<caption> XXX  data table</caption>

<tr>
<td align="center" width="10%"><b>id & labels</b></td>
<td align="center" width="10%"><b>predictions</b></td>
<td align="center" width="20%"><b>Text</b></td>
<td align="center" width="15%"><b>TP</b></td>
<td align="center" width="15%"><b>FP</b></td>
<td align="center" width="15%"><b>FN</b></td>
<td align="center" width="15%" style="display:none"><b>TN</b></td>
</tr>
'''

postBody = '''
</table>
'''

scripts = '''
<script>
function refreshTable() {

	var tp = document.getElementById("TP");
	var fp = document.getElementById("FP");
	var fn = document.getElementById("FN");
	var tn = document.getElementById("TN");

	var table = document.getElementById("mytable");
	var rows = table.getElementsByTagName("tr");
	for (var row = 0; row < rows.length; row++) {
		var cols = rows[row].children;

		if (tp.checked && cols[3].innerHTML != "" || 
			fp.checked && cols[4].innerHTML != "" || 
			fn.checked && cols[5].innerHTML != "" || 
			tn.checked && cols[6].innerHTML != "") {
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
}

function highlightText(rownum) {

	var table = document.getElementById("mytable");
	var rows = table.getElementsByTagName('tr');

	var cols = rows[rownum[0]].children;

	var cell = cols[2];

	var text = cell.innerHTML;
	text = text.replace(/<font color="red">/g, "");
	text = text.replace(/<\/font>/g, "");
	var words = text.split(" ");

	for(var j = 1; j < rownum.length; j++) {
		pos = rownum[j];
		words[pos] = "<font color='red'>"+words[pos]+"</font>";
	}

	cell.innerHTML = words.join(" ");

}
</script>
'''

es = Elasticsearch("localhost:9200", timeout=600, max_retries=10, revival_delay=0)


def main():
    # usage: myprog json_file
    if len(sys.argv) > 1:
        jsonFile = sys.argv[1]
    else:
        print "usage: myprog json_file"
        jsonFile = "train_prediction_analysis.json"
    start = time.time()
    parse(jsonFile)
    end = time.time()
    print "parsing cost time ", start-end, " seconds"


if __name__ == '__main__':
    main()





