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
        output.write("<li style='display:none'>")
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
    # default rule number is 6
    output.write("<ul class='ruleList'>")
    output.write("<li>prior: %.2f</li>" % clas["rules"][0]["score"])
    for i in range(1, min(7,len(clas["rules"]))):
        writeRule(docId, line_count, i, output, clas["rules"][i], True)
    for i in range(7, len(clas["rules"])):
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
    for i in range(0,classNumber):
        if (i not in labelSet) and (i not in predictionSet):
            writeClass(row["id"],line_count, output, row["classScoreCalculations"][i])
    output.write("</td>")


def writeTable(output, data):
    line_count = 0
    for row in data:
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
        res = es.get(index=esIndex,
                     doc_type="document",
                     id=row["id"])
        output.write(res["_source"]["body"].encode('utf-8'))
        output.write("</td>")
        # column 4 - 7 TP FP FN TN columns
        writeTableTFPNColumns(row, line_count, output)
        # finish row
        output.write("</tr>")
        # test break after first line
        if(line_count % 100 == 0):
            print "Current parsing ID: ", line_count
            break


def writeBody(output, data):
    output.write("<body>")
    output.write(preBody)
    writeTable(output, data)
    output.write(postBody)
    output.write(scripts)
    output.write("</body>")


def parse(input_json_file):
    # read input
    input_json = open(input_json_file, "r")
    data = json.load(input_json)
    print "Json load successfully.\nStart Parsing..."
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
#mytable{
border: 1px solid black;
border-collapse: collapse;
}
#optionTable {
border : 1px solid black;
border-collapse: collapse;
float : center;
width : 30%;
}
.ruleList {
list-style-type: square;
}
</style>
</head>
'''


preBody = '''
<br>
<table id='optionTable'>
<tr><th><br> Data Viewer Options: </th></tr>
<tr><td><br>
<p style="text-indent: 1em;">Max number of rules per class:&nbsp;&nbsp;
<input id="ruleNum" type="number" name="ruleNum" value="6" min="1"></p>
</td></tr><tr><td><br>
<p style="text-indent: 1em;">Select the fields to display
<input id="TP" type="checkbox" name="TP" value="TP" checked>TP
<input id="FP" type="checkbox" name="FP" value="FP" checked>FP
<input id="FN" type="checkbox" name="FN" value="FN" checked>FN
<input id="TN" type="checkbox" name="TN" value="TN" >TN</p>
</td></tr><tr><td><br><br>
<center><button onclick="refreshTable()">Submit</button></center>
<br></td></tr>
</table>

<br><br>

<p>Rules and classes:</p>
<p>(Press any rules would highlight matched keywords in the text if they exist)</p>
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
	text = text.replace(/<font color=/g, "<font-color=");
	var words = text.split(" ");

	if(isNewHighLight(words, rownum)) {

		var colors = ["red", "Magenta", "lime", "blue", "GreenYellow", "LightPink", "orange", "yellow", "LightSeaGreen", "Orchid"];
		var pickColor = pickNewColor(text, colors);

		for(var i = 1; i < rownum.length; i++) {
			var pos = rownum[i];
			words[pos] = words[pos].replace(/(<([^>]+)>)/ig,"");
			words[pos] = "<font color='" + pickColor + "'>"+words[pos]+"</font>";
		}
	}
	else {
		for(var i = 1; i < rownum.length; i++) {
			var pos = rownum[i];
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

</script>
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




