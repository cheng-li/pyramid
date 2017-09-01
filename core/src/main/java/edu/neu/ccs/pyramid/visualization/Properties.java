package edu.neu.ccs.pyramid.visualization;

/**
 * Created by shikhar on 6/29/17.
 */
public class Properties {

    /*   File names   */
    public static final String CONFIG_FILE_NAME = "data_config.json";
    public static final String DATA_FILE_NAME = "data_info.json";
    public static final String MODEL_FILE_NAME = "model_config.json";

    /*   HTML   */
    public static final String PRE_HTML =
            "<html>\n" +
                    "    <head>\n" +
                    "        <script src=\"https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js\"></script>\n" +
                    "        <!-- <script src=\"./js/jquery.min.js\"></script> -->\n" +
                    "    </head>\n" +
                    "    <body><br>\n" +
                    "        <table id='optionTable'  style='width:55%'>\n" +
                    "            <tr><th><br> Data Viewer Options: \n" +
                    "            </th></tr>\n" +
                    "            <tr><td><br>\n" +
                    "                <p style=\"text-indent: 1em;\">Max number of rules per class:&nbsp;&nbsp;\n" +
                    "                <input id=\"ruleNum\" type=\"number\" name=\"ruleNum\" value=\"12\" min=\"1\" style='width:3em' onchange=\"refresh()\"></p>\n" +
                    "            </td></tr>\n" +
                    "            <tr><td>\n" +
                    "                <p style=\"text-indent: 1em;\">Number of labels in predictedRanking:&nbsp;&nbsp;\n" +
                    "                <select id=\"numOfLabels\" onchange=\"refresh()\">\n" +
                    "                    <option value=5>TOP 5\n" +
                    "                    <option value=10 >TOP 10\n" +
                    "                    <option value=15 >TOP 15\n" +
                    "                    <option value=-1 >ALL\n" +
                    "                    <option value=-2 selected>Show Each Until Only TN\n" +
                    "                </select>\n" +
                    "            </td></tr>\n" +
                    "            <tr><td>\n" +
                    "                <p style=\"text-indent: 1em;\">Select the fields to display\n" +
                    "                <input id=\"TP\" type=\"checkbox\" name=\"TP\" value=\"TP\" checked>TP\n" +
                    "                <input id=\"FP\" type=\"checkbox\" name=\"FP\" value=\"FP\" checked>FP\n" +
                    "                <input id=\"FN\" type=\"checkbox\" name=\"FN\" value=\"FN\" checked>FN\n" +
                    "                <input id=\"TN\" type=\"checkbox\" name=\"TN\" value=\"TN\">TN</p>\n" +
                    "            </td></tr>\n" +
                    "            <tr><td>\n" +
                    "                <p style=\"text-indent: 1em;\">Sort Docs:\n" +
                    "                <input id=\"Confide\" type=\"radio\" name=\"sd\" value=\"confide\">Confide\n" +
                    "                <input id=\"Mistake\" type=\"radio\" name=\"sd\" value=\"mistake\" checked>Mistake\n" +
                    "                <input id=\"test\" type=\"radio\" name=\"sd\" value=\"test\">By Id\n" +
                    "            </td></tr>\n" +
                    "            <tr><td>\n" +
                    "                <p style=\"text-indent: 1em;\">Sort Rules:\n" +
                    "                <input id=\"abs\" type=\"radio\" name=\"sr\" value=\"abs\">Abs Descending\n" +
                    "                <input id=\"ascending\" type=\"radio\" name=\"sr\" value=\"ascending\">Ascending\n" +
                    "                <input id=\"descending\" type=\"radio\" name=\"sr\" value=\"descending\">Descending\n" +
                    "                <input id=\"anti\" type=\"radio\" name=\"sr\" value=\"anti\" checked>Anticorrelation\n" +
                    "            </td></tr>\n" +
                    "            <tr><td>\n" +
                    "                <p style=\"text-indent: 1em;\">Rule display:\n" +
                    "                <input id=\"details\" type=\"checkbox\" name=\"details\" value=\"details\">Details\n" +
                    "            </td></tr>\n" +
                    "            <tr><td>\n" +
                    "                <p style=\"text-indent: 1em;\">\n" +
                    "                <a href=\"top_features.html\" target=\"_blank\">Top Features</a>\n" +
                    "                <a href=\"metadata.html\" target=\"_blank\">Metadata</a>\n" +
                    "                <a href=\"individual_performance.html\" target=\"_blank\">Performance</a>\n" +
                    "            </td></tr>\n" +
                    "            <tr><td>\n" +
                    "                <center><button id=\"createFile\">Create New HTML</button> \n" +
                    "                <a download=\"new.html\" id=\"downloadlink\" style=\"display: none\">Download</a></center><br>\n" +
                    "            </td></tr>\n" +
                    "        </table><br><br>\n" +
                    "\n" +
                    "        <p>Feedbacks:</p>\n" +
                    "        <table id=\"feedbackTable\" border=1>\n" +
                    "            <thead><tr>\n" +
                    "                <td align=\"center\"><b>failure</b></td>\n" +
                    "                <td align=\"center\"><b>incomplete</b></td>\n" +
                    "                <td align=\"center\"><b>bad_rule</b></td>\n" +
                    "                <td align=\"center\"><b>bad_matching</b></td>\n" +
                    "            </tr></thead>\n" +
                    "            <tbody id=\"data-feedbackTable\"></tbody>\n" +
                    "        </table>\n" +
                    "\n" +
                    "        <p>Rules and classes:</p>\n" +
                    "        <p>(Press any rules would highlight matched keywords in the text if they exist)</p>\n" +
                    "\n" +
                    "        <table id=\"mytable\" border=1  align=\"center\" style=\"width:100%\">\n" +
                    "            <caption> Report </caption>\n" +
                    "                <thead><tr>\n" +
                    "                    <td align=\"center\" width=\"20%\"><b>id & labels</b></td>\n" +
                    "                    <td align=\"center\" width=\"20%\"><b>Text</b></td>\n" +
                    "                    <td align=\"center\" width=\"20%\"><b>TP</b></td>\n" +
                    "                    <td align=\"center\" width=\"20%\"><b>FP</b></td>\n" +
                    "                    <td align=\"center\" width=\"20%\"><b>FN</b></td>\n" +
                    "                    <td align=\"center\" width=\"20%\"><b>TN</b></td>\n" +
                    "                </tr></thead>\n" +
                    "            <tbody id=\"data-table\"></tbody>\n" +
                    "        </table>\n" +
                    "\n" +
                    "        <script>\n" +
                    "            function changeFeedback(row) {\n" +
                    "                document.getElementById('feedback' + row).style.display = \"block\";\n" +
                    "\n" +
                    "                refreshTable(viewOptions())\n" +
                    "            }\n" +
                    "\n" +
                    "            function download() {\n" +
                    "                feedbacks = []\n" +
                    "\n" +
                    "                var table = document.getElementById(\"mytable\");\n" +
                    "                var rows = table.getElementsByTagName(\"tr\");\n" +
                    "                for (var row = 0; row < rows.length; row++) {\n" +
                    "                    var feedback = {}\n" +
                    "                    myselect = document.getElementById('sel' + row)\n" +
                    "                    if (myselect != null && (option = myselect.options[myselect.selectedIndex].value) != 'none') {\n" +
                    "                        key = document.getElementsByTagName('pre')[row].firstChild.data.replace (/  +$/, '')\n" +
                    "                        feedback[key] = {'option:': option, 'text': document.getElementById('feedback' + row).value}\n" +
                    "                        feedbacks.push(feedback)\n" +
                    "                    }\n" +
                    "                }\n" +
                    "                var text = JSON.stringify(feedbacks); \n" +
                    "\n" +
                    "                url = window.location.href\n" +
                    "                var pom = document.createElement('a');\n" +
                    "                pom.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));\n" +
                    "                pom.setAttribute('download', 'Viewer_with_Feedbacks.html');\n" +
                    "\n" +
                    "                pom.style.display = 'none';\n" +
                    "                document.body.appendChild(pom);\n" +
                    "\n" +
                    "                pom.click();\n" +
                    "\n" +
                    "                document.body.removeChild(pom);\n" +
                    "            }\n" +
                    "\n" +
                    "            function convertHighlightsIntoPositions(highlights) {\n" +
                    "                var colors = [\"red\", \"Magenta\", \"lime\", \"blue\", \"GreenYellow\", \"LightPink\", \n" +
                    "                        \"orange\", \"yellow\", \"LightSeaGreen\", \"Orchid\"];\n" +
                    "                positions = {}\n" +
                    "\n" +
                    "                for (var i = 0; i < highlights.length; i++) {\n" +
                    "                    color = colors[i % colors.length]\n" +
                    "                    poses = highlights[i]\n" +
                    "                    for (var j = 0; j < poses.length; j++) {\n" +
                    "                        positions[poses[j][0]] = {'end': poses[j][1], 'color': color}\n" +
                    "                    }\n" +
                    "                }\n" +
                    "\n" +
                    "                return positions\n" +
                    "            }\n" +
                    "\n" +
                    "            function convertArrayToString(arr) {\n" +
                    "                str = \"\"\n" +
                    "\n" +
                    "                for (j = 0; j < arr.length; j++) {\n" +
                    "                    a = arr[j]\n" +
                    "                    str += a.toString()\n" +
                    "                    if (j != arr.length - 1) {\n" +
                    "                        str += \" \"\n" +
                    "                    }\n" +
                    "                }\n" +
                    "                return str\n" +
                    "            }\n" +
                    "\n" +
                    "            function convertStringToArray(str) {\n" +
                    "                arr = []\n" +
                    "                if (str == \"\") {\n" +
                    "                    return []\n" +
                    "                }\n" +
                    "\n" +
                    "                splits = str.split(\" \")\n" +
                    "                for (i = 0; i < splits.length; i++) {\n" +
                    "                    a = []\n" +
                    "                    pairs = splits[i].split(\",\")\n" +
                    "                    for (j = 0; j < pairs.length; j += 2) {\n" +
                    "                        a.push([pairs[j], pairs[j + 1]])\n" +
                    "                    }\n" +
                    "                    arr.push(a)\n" +
                    "                }\n" +
                    "\n" +
                    "                return arr\n" +
                    "            }\n" +
                    "\n" +
                    "            function areSamePoses(pos1, pos2) {\n" +
                    "                if (pos1.length != pos2.length) {\n" +
                    "                    return false\n" +
                    "                }\n" +
                    "\n" +
                    "                for (k = 0; k < pos1.length; k++) {\n" +
                    "                    if (pos1[k][0] != pos2[k][0] || pos1[k][1] != pos2[k][1]) {\n" +
                    "                        return false\n" +
                    "                    }\n" +
                    "                }\n" +
                    "\n" +
                    "                return true\n" +
                    "            }\n" +
                    "\n" +
                    "            function indexOfHighlights(poses, highlights) {\n" +
                    "                for (i = 0; i < highlights.length; i++) {\n" +
                    "                    highlight = highlights[i]\n" +
                    "                    if (areSamePoses(poses, highlight)) {\n" +
                    "                        return i\n" +
                    "                    }\n" +
                    "                }\n" +
                    "                return -1\n" +
                    "            }\n" +
                    "\n" +
                    "            function highlightText(poses, rowNum, field) {\n" +
                    "                var table = document.getElementById(\"mytable\");\n" +
                    "                var rows = table.getElementsByTagName('tr');\n" +
                    "                var cols = rows[rowNum].children;\n" +
                    "                var cell = cols[1];\n" +
                    "                var tagName = \"#highlights\" + (rowNum - 1)\n" +
                    "                var highlights = $(tagName).data('data');\n" +
                    "\n" +
                    "                if (rowNum.length == 0) {\n" +
                    "                    return\n" +
                    "                }\n" +
                    "\n" +
                    "                newText = \"\"\n" +
                    "                for (key in highlights) {\n" +
                    "                    hs = highlights[key]\n" +
                    "                    if (key == field) {\n" +
                    "                        if ((index = indexOfHighlights(poses, hs)) != -1) {\n" +
                    "                            hs.splice(index, 1)\n" +
                    "                        }\n" +
                    "                        else {\n" +
                    "                            hs.push(poses)\n" +
                    "                        }\n" +
                    "                    }\n" +
                    "\n" +
                    "                    positions = convertHighlightsIntoPositions(hs)\n" +
                    "                    keys = Object.keys(positions)\n" +
                    "                    var text = document.getElementById(key + (rowNum - 1)).innerHTML;\n" +
                    "                    newText += key + \":<br>\"\n" +
                    "                    start = 0\n" +
                    "                    for (var i = 0; i < keys.length; i++) {\n" +
                    "                        index = keys[i]\n" +
                    "                        end = index\n" +
                    "                        newText += text.substring(start, end) + \n" +
                    "                            text.substring(end, positions[index]['end']).fontcolor(positions[index]['color'])\n" +
                    "                        start = positions[index]['end']\n" +
                    "                    }\n" +
                    "                    newText += text.substring(start, text.length) + \"<br>\"\n" +
                    "                }\n" +
                    "\n" +
                    "                cell.innerHTML = newText\n" +
                    "                $(tagName).data('data', highlights);\n" +
                    "            }\n" +
                    "\n" +
                    "            function createOption(value, isSelected) {\n" +
                    "                if (isSelected) {\n" +
                    "                    return \"<option value=\" + value + \" selected>\" + value + \"</option>\"\n" +
                    "                } else {\n" +
                    "                    return \"<option value=\" + value + \">\" + value + \"</option>\"\n" +
                    "                }\n" +
                    "            }\n" +
                    "\n" +
                    "            function createTextarea(i, display, text) {\n" +
                    "                if (display) {\n" +
                    "                    return \"<textarea id=feedback\" + i + \" maxlength='512' class='mytext' cols='40' rows='5'>\" +\n" +
                    "                        text + \"</textarea>\"\n" +
                    "                } else {\n" +
                    "                    return \"<textarea id=feedback\" + i + \" maxlength='512' class='mytext' cols='40' rows='5' style='display:none'>\" +\n" +
                    "                        text + \"</textarea>\" \n" +
                    "                }\n" +
                    "            }\n" +
                    "\n" +
                    "            function displayFeedback(i, sel, text) {\n" +
                    "                return \"<select onchange='changeFeedback(\" + i + \")' id=sel\" + i + \" >\" + \n" +
                    "                createOption('none', sel=='none') + \n" +
                    "                createOption('failure', sel=='failure') + \n" +
                    "                createOption('incomplete', sel=='incomplete') + \n" +
                    "                createOption('bad_rule', sel=='bad_rule') + \n" +
                    "                createOption('bad_matching', sel=='bad_matching') + \n" +
                    "                \"</select>\" + \n" +
                    "                createTextarea(i, sel!='none', text)\n" +
                    "                \n" +
                    "            }\n" +
                    "\n" +
                    "            function createNewHTML() {\n" +
                    "                //var textFile = null\n" +
                    "                var create = document.getElementById('createFile')\n" +
                    "                create.addEventListener('click', function () {\n" +
                    "                    feedbacks = {}\n" +
                    "                    startStr = \"<script id=\\\"raw-data\\\" type=\\\"application/json\\\">\"\n" +
                    "                    endStr = \"<\\/script><\\/body><\\/html>\"\n" +
                    "\n" +
                    "                    var table = document.getElementById(\"mytable\");\n" +
                    "                    var rows = table.getElementsByTagName(\"tr\");\n" +
                    "                    for (var row = 0; row < rows.length; row++) {\n" +
                    "                        myselect = document.getElementById('sel' + row)\n" +
                    "                        if (myselect != null && (option = myselect.options[myselect.selectedIndex].value) != 'none') {\n" +
                    "                            key = parseInt(document.getElementsByTagName('pre')[row].firstChild.data.replace (/  +$/, ''))\n" +
                    "                            feedbacks[key] = {'option': option, 'text': document.getElementById('feedback' + row).value}\n" +
                    "                        }\n" +
                    "                    }\n" +
                    "\n" +
                    "                    data = dataFromJson()\n" +
                    "                    data.forEach(function(row) {\n" +
                    "                        key = parseInt(row.idlabels.id)\n" +
                    "                        if (feedbacks[key] != undefined) {\n" +
                    "                            row.idlabels.feedbackSelect = feedbacks[key]['option']\n" +
                    "                            row.idlabels.feedbackText = feedbacks[key]['text']\n" +
                    "                        }\n" +
                    "                        else {\n" +
                    "                            row.idlabels.feedbackSelect = 'none'\n" +
                    "                        }\n" +
                    "                    })\n" +
                    "                    dataJson = JSON.stringify(data);\n" +
                    "                    text = document.documentElement.innerHTML\n" +
                    "                    finalHTML = text.substring(0, text.indexOf(startStr)) + startStr + dataJson + endStr\n" +
                    "                    var link = document.getElementById('downloadlink');\n" +
                    "                    link.href = makeTextFile(finalHTML);\n" +
                    "                    link.style.display = 'block';\n" +
                    "                }, false);\n" +
                    "            }\n" +
                    "\n" +
                    "            function refresh() {\n" +
                    "                var displayOptions = viewOptions()\n" +
                    "                render(sortByViewOptions(dataFromJson(), displayOptions), displayOptions)\n" +
                    "            }\n" +
                    "\n" +
                    "            function generateFeedbackDataTable(data) {\n" +
                    "                feedbacks = {'failure':0, 'incomplete':0, 'bad_rule':0, 'bad_matching':0}\n" +
                    "                var $body = $('#data-feedbackTable')\n" +
                    "                $body.empty()\n" +
                    "\n" +
                    "                data.forEach(function (row, i) {\n" +
                    "                    feedbacks[row.idlabels.feedbackSelect] += 1\n" +
                    "                })\n" +
                    "\n" +
                    "                var html = ''\n" +
                    "                html += '<tr>' +\n" +
                    "                    \"<td style='vertical-align:top;text-align:left;'>\" + feedbacks['failure'] + '</td>' +\n" +
                    "                    \"<td style='vertical-align:top;text-align:left;'>\" + feedbacks['incomplete'] + '</td>' +\n" +
                    "                    \"<td style='vertical-align:top;text-align:left;'>\" + feedbacks['bad_rule'] + '</td>' +\n" +
                    "                    \"<td style='vertical-align:top;text-align:left;'>\" + feedbacks['bad_matching'] + '</td>' +\n" +
                    "                    +'</tr>'\n" +
                    "\n" +
                    "                $body.append(html)\n" +
                    "            }\n" +
                    "\n" +
                    "            function getLabelColor(type) {\n" +
                    "                if (type == \"TP\") {\n" +
                    "                    return \"green\"\n" +
                    "                } \n" +
                    "                else if (type == \"FN\") {\n" +
                    "                    return \"green\"\n" +
                    "                }\n" +
                    "                else if (type == \"TN\") {\n" +
                    "                    return \"black\"\n" +
                    "                }\n" +
                    "                else if (type == \"FP\") {\n" +
                    "                    return \"red\"\n" +
                    "                }\n" +
                    "                else {\n" +
                    "                    return \"\"\n" +
                    "                }\n" +
                    "            }\n" +
                    "\n" +
                    "            function includesLabel(clas, label) {\n" +
                    "                for (i = 0; i < clas.length; i++) {\n" +
                    "                    if (clas[i].name == label) {\n" +
                    "                        return true\n" +
                    "                    }\n" +
                    "                }\n" +
                    "                return false\n" +
                    "            }  \n" +
                    "\n" +
                    "            function displayPredictedRanking(row, displayOptions) {\n" +
                    "                numOfLabels = 0\n" +
                    "                predictedRanking = row.predictedRanking\n" +
                    "                var split = false\n" +
                    "\n" +
                    "                if (displayOptions.numOfLabels == -1) {\n" +
                    "                    numOfLabels = predictedRanking.length\n" +
                    "                } else if (displayOptions.numOfLabels == -2) {\n" +
                    "                    for (i = 0; i < predictedRanking.length; i++) {\n" +
                    "                        if (predictedRanking[i].type != \"\") {\n" +
                    "                            numOfLabels = i + 1\n" +
                    "                        }\n" +
                    "                    }\n" +
                    "                } else {\n" +
                    "                    numOfLabels = displayOptions.numOfLabels\n" +
                    "                }\n" +
                    "\n" +
                    "                if (numOfLabels > 0) {\n" +
                    "                    return serialize(predictedRanking.slice(0, numOfLabels), function (lb) {\n" +
                    "                            var str = ''\n" +
                    "                            text = lb.className + '(' + lb.prob.toFixed(2) + ')'\n" +
                    "                            if (split == false && lb.prob.toFixed(2) < 0.5) {\n" +
                    "                                split = true\n" +
                    "                            }\n" +
                    "                            if (lb.type == \"TP\" || lb.type == \"FP\") {\n" +
                    "                                str += '<li><span style=\"background-color:lightGray\">' + text.fontcolor(getLabelColor(lb.type)) + '</span></li>'\n" +
                    "                            } else {\n" +
                    "                                str += '<li style=\"list-style-type:none;\">&nbsp&nbsp&nbsp' + text.fontcolor(getLabelColor(lb.type)) + '</li>'\n" +
                    "                            }\n" +
                    "\n" +
                    "                            return str\n" +
                    "                    }) \n" +
                    "                } else {\n" +
                    "                    return \"\"\n" +
                    "                }\n" +
                    "            }\n" +
                    "\n" +
                    "            function labelsToString(labels) {\n" +
                    "                var str = ''\n" +
                    "\n" +
                    "                if (labels.length == 0) {\n" +
                    "                    return str\n" +
                    "                }\n" +
                    "\n" +
                    "                str += labels[0]\n" +
                    "\n" +
                    "                for (var i = 1; i < labels.length; i++) {\n" +
                    "                    str += \"|\" + labels[i]\n" +
                    "                }\n" +
                    "\n" +
                    "                return str\n" +
                    "            }\n" +
                    "\n" +
                    "            function displayLabelSetRanking(row) {\n" +
                    "                predictedLabelSetRanking = row.predictedLabelSetRanking\n" +
                    "                str = ''\n" +
                    "\n" +
                    "                for (var i = 0; i < predictedLabelSetRanking.length; i++) {\n" +
                    "                    labels = predictedLabelSetRanking[i]\n" +
                    "                    if (labels.labels.length == 0) {\n" +
                    "                        temp = \"EMPTY_SET\".fontcolor(\"black\")\n" +
                    "                        temp += \"(\" + labels.probability.toFixed(2) +\")\"\n" +
                    "                    } else {\n" +
                    "                        temp = labels.labels[0].fontcolor(getLabelColor(labels.types[0]))\n" +
                    "                        for (var j = 1; j < labels.labels.length; j++) {\n" +
                    "                            temp += \" | \" + labels.labels[j].fontcolor(getLabelColor(labels.types[j]))\n" +
                    "                        }\n" +
                    "\n" +
                    "                        temp += '(' + labels.probability.toFixed(2)  + ')'\n" +
                    "                    }\n" +
                    "\n" +
                    "                    if (i == 0) {\n" +
                    "                        temp = '<span style=\"background-color:lightGray\">' + temp + '</span>'\n" +
                    "                    }\n" +
                    "\n" +
                    "                    temp = '<li>' + temp + '</li>'\n" +
                    "                    str += temp\n" +
                    "                }\n" +
                    "\n" +
                    "                return str\n" +
                    "            }\n" +
                    "\n" +
                    "            function displayText(text) {\n" +
                    "                keys = Object.keys(text)\n" +
                    "                str = ''\n" +
                    "                for (var i = 0; i < keys.length; i++) {\n" +
                    "                    key = keys[i]\n" +
                    "                    str += key + \":<br>\" + text[key] + \"<br>\"\n" +
                    "                }\n" +
                    "\n" +
                    "                return str\n" +
                    "            }\n" +
                    "\n" +
                    "            function storeOrigText(text, index) {\n" +
                    "                str = \"\"\n" +
                    "                keys = Object.keys(text)\n" +
                    "                for (var i = 0; i < keys.length; i++) {\n" +
                    "                    str += \"<pre id=\" + keys[i] + index + \" style='display:none'>\" + text[keys[i]] + '</pre>'\n" +
                    "                }\n" +
                    "\n" +
                    "                return str\n" +
                    "            }\n" +
                    "\n" +
                    "            function displayOthers(others) {\n" +
                    "                str = ''\n" +
                    "\n" +
                    "                keys = Object.keys(others)\n" +
                    "                for (i = 0; i < keys.length; i++) {\n" +
                    "                    key = keys[i]\n" +
                    "                    str += \"<br><b>\" + key + \"</b>: \" + others[key]\n" +
                    "                }\n" +
                    "\n" +
                    "                return str\n" +
                    "            }\n" +
                    "\n" +
                    "            function initialHighlights(data) {\n" +
                    "                data.forEach(function (row, i) {\n" +
                    "                    keys = Object.keys(row.text)\n" +
                    "                    highlights = {}\n" +
                    "                    for (j = 0; j < keys.length; j++) {\n" +
                    "                        highlights[keys[j]] = []\n" +
                    "                    }\n" +
                    "                    $(\"#highlights\" + i).data('data', highlights);\n" +
                    "                })\n" +
                    "            }\n" +
                    "\n" +
                    "            function render(data, displayOptions) {\n" +
                    "                generateFeedbackDataTable(data)\n" +
                    "\n" +
                    "                var $body = $('#data-table')\n" +
                    "                $body.empty()\n" +
                    "                var html = ''\n" +
                    "                data.forEach(function (row, i) {\n" +
                    "                    var labels = ''\n" +
                    "\n" +
                    "                    html += '<tr>' +\n" +
                    "                        \"<td style='vertical-align:top;text-align:left;' width='5%'>\" + \n" +
                    "                        \"<pre id='labelId\" + i + \"' style='display:none'>\" + row.idlabels.id + '</pre>' +\n" +
                    "                        \"<input id='highlights\" + i + \"' style='display:none' value=''>\" +\n" +
                    "                        storeOrigText(row.text, i) +\n" +
                    "                        \"<b>ID:</b>&nbsp\" + row.idlabels.id + \n" +
                    "                        displayOthers(row.others) + \n" +
                    "                        '<br><b>Labels</b>:' +  \n" +
                    "                        serialize(row.idlabels.internalLabels, function (lb) {\n" +
                    "                            var str = ''\n" +
                    "                            for (var k in lb) {\n" +
                    "                                str += '<li>' + lb[k] + '</li>'\n" +
                    "                            }\n" +
                    "                            return str\n" +
                    "                         }) + \n" +
                    "                        '<br><b>Predictions</b>:' + \n" +
                    "                        serialize(row.idlabels.predictions, function (lb) {\n" +
                    "                            var str = ''\n" +
                    "                            for (var k in lb) {\n" +
                    "                                str += '<li>' + lb[k] + '</li>'\n" +
                    "                            }\n" +
                    "                            return str\n" +
                    "                        }) +\n" +
                    "                        '<br><b>Label&nbspRanking</b>:' +\n" +
                    "                        displayPredictedRanking(row, displayOptions) + \n" +
                    "                        \"<br><b>AP:&nbsp</b>\" + row.idlabels.ap +\n" +
                    "                        \"<br><b>RankOfFullRecall:&nbsp</b>\" + row.idlabels.rankoffullrecall + \"<br>\" +\n" +
                    "                        '<br><b>Label&nbspSet&nbspRanking</b>:' +\n" +
                    "                        displayLabelSetRanking(row, displayOptions) + \n" +
                    "                        \"<br><b>Overlap:&nbsp</b>\" + row.idlabels.overlap +\n" +
                    "                        \"<br><b>Precision:&nbsp</b>\" + row.idlabels.precision +\n" +
                    "                        \"<br><b>Recall:&nbsp</b>\" + row.idlabels.recall +\n" +
                    "                        '<br><br><br><b>Feedback</b>:' +\n" +
                    "                        displayFeedback(i, row.idlabels.feedbackSelect, row.idlabels.feedbackText) +\n" +
                    "                        '</td>' +\n" +
                    "                        \"<td style='vertical-align:top;text-align:left;'>\" + \n" +
                    "                        displayText(row.text) +\n" +
                    "                        '</td>' +\n" +
                    "                        displayClass(row.TP, displayOptions, i) +\n" +
                    "                        displayClass(row.FP, displayOptions, i) +\n" +
                    "                        displayClass(row.FN, displayOptions, i) +\n" +
                    "                        displayClass(row.TN, displayOptions, i) +\n" +
                    "                        '</tr>'\n" +
                    "\n" +
                    "\n" +
                    "                })\n" +
                    "\n" +
                    "                $body.append(html)\n" +
                    "                initialHighlights(data)\n" +
                    "                refreshTable(displayOptions)\n" +
                    "                createNewHTML()\n" +
                    "            }\n" +
                    "\n" +
                    "\n" +
                    "            function createRuleDetails(rule, displayOptions, rowNum) {\n" +
                    "                str = \"\"\n" +
                    "                score = ''\n" +
                    "                if (rule.score >= 0) {\n" +
                    "                    score += \": +\" + Math.abs(rule.score).toExponential(2)\n" +
                    "                } else {\n" +
                    "                    score += \": -\" + Math.abs(rule.score).toExponential(2)\n" +
                    "                }\n" +
                    "                str += '<li>' + serialize(rule['checks'], function (check) {\n" +
                    "                            style = \"style='color:#0000FF; margin:0px; padding:0px;' onclick='highlightText(\" + check.highlights + \", \" + \n" +
                    "                                (rowNum + 1) + \", \\\"\" + check.field + \"\\\")'\"\n" +
                    "                            //alert(Object.keys(check))\n" +
                    "                            if ('ngram' in check) {\n" +
                    "                                str = '<p ' + style + '>' + check.ngram + ' [' + check.value.toExponential(2) + check.relation + \n" +
                    "                                check.threshold.toExponential(2) +']'\n" +
                    "                                if (displayOptions.details) {\n" +
                    "                                    str += 'index=' + check.index + ' field=' + check.field + ' slop:' + check.slop\n" +
                    "                                }\n" +
                    "                            }\n" +
                    "                            else {\n" +
                    "                                str = '<p ' + style + '>' + check.name + ' [' + check.value.toExponential(2) + check.relation + \n" +
                    "                                check.threshold.toExponential(2) +']'\n" +
                    "                                if (displayOptions.details) {\n" +
                    "                                    str += 'index=' + check.index\n" +
                    "                                }\n" +
                    "                            }\n" +
                    "                            return str\n" +
                    "                        }) + '</li>'\n" +
                    "\n" +
                    "\n" +
                    "                return str + score\n" +
                    "            }\n" +
                    "\n" +
                    "            // BW: highlight all rules by clicked label.\n" +
                    "            // function highlightAll(rules, rowNum) {\n" +
                    "            //    console.log(rules);\n" +
                    "            //    console.log(rowNum);\n" +
                    "            //    for (i=0; i<rules.length; i++) {\n" +
                    "            //        rule = rules[i]\n" +
                    "            //        for (j=0; j<rule['checks'].length; j++) {\n" +
                    "            //            check = rule['checks'][j]\n" +
                    "            //            highlightText(check.highlights, (rowNum+1), check.field)\n" +
                    "            //        }\n" +
                    "            //    }\n" +
                    "            //}\n" +
                    "\n" +
                    "            //function printTest(rules,rowNum){\n" +
                    "            //    console.log(rules);\n" +
                    "            //    console.log(rowNum)\n" +
                    "            //}\n" +
                    "\n" +
                    "            function displayClass(clas, displayOptions, rowNum) {\n" +
                    "                str = \"\"\n" +
                    "                str += \"<td style='vertical-align:top;text-align:left;'>\" +\n" +
                    "                        serialize(clas, function (lb, i) {\n" +
                    "                            str = \"\"\n" +
                    "                            prior = \"\"\n" +
                    "                            if (lb.prior != undefined) {\n" +
                    "                                prior = '<li>prior: ' + lb.prior.toExponential(2) + '</li>'\n" +
                    "                            }\n" +
                    "                            if (i > 0) {\n" +
                    "                                str += '<hr>'\n" +
                    "                            }\n" +
                    "                            // BW: add clickable style to label\n" +
                    "                            style = \"style='color:#00FF00; margin:0px; padding:0px;' onclick='highlightText(\" + lb.allPos + \",\" + \n" +
                    "                                (rowNum + 1) + \", \\\"\" + lb.field + \"\\\")'\"\n" +
                    "                            str += '<p ' + style + '>' + lb.name + '</p>' + '<br><br>classProbability: ' + \n" +
                    "                            lb.classProbability.toExponential(2) + '<br><br>totalScore: ' + \n" +
                    "                            lb.totalScore.toExponential(2) + '<br><ul>' + \n" +
                    "                            prior +\n" +
                    "                            serialize(lb.rules, function (rule, i) {\n" +
                    "                                if (i >= displayOptions.ruleNum) {\n" +
                    "                                    return \"\"\n" +
                    "                                }  \n" +
                    "                                return createRuleDetails(rule, displayOptions, rowNum)\n" +
                    "                            })+ '</ul>'\n" +
                    "                            return str\n" +
                    "                        }) + '</td>'\n" +
                    "                return str\n" +
                    "            }\n" +
                    "\n" +
                    "            function serialize(a, cb) {\n" +
                    "                var str = ''\n" +
                    "\n" +
                    "                if (a == undefined) {\n" +
                    "                    return str\n" +
                    "                }\n" +
                    "\n" +
                    "                a.forEach(function (obj, i) {\n" +
                    "                    if (cb) {\n" +
                    "                        str += cb(obj, i)\n" +
                    "                    } else {\n" +
                    "                        str += obj\n" +
                    "                    }\n" +
                    "                })\n" +
                    "                return str\n" +
                    "            }\n" +
                    "\n" +
                    "            function viewOptions() {\n" +
                    "                var displayOptions = {}\n" +
                    "\n" +
                    "                displayOptions.ruleNum = parseInt($('#ruleNum').val())\n" +
                    "                displayOptions.fields = []\n" +
                    "                if ($('#TP').prop('checked'))\n" +
                    "                    displayOptions.fields.push('TP')\n" +
                    "                if ($('#FP').prop('checked'))\n" +
                    "                    displayOptions.fields.push('FP')\n" +
                    "                if ($('#FN').prop('checked'))\n" +
                    "                    displayOptions.fields.push('FN')\n" +
                    "                if ($('#TN').prop('checked'))\n" +
                    "                    displayOptions.fields.push('TN')\n" +
                    "\n" +
                    "                displayOptions.sortDocs = $('input[name=sd]:checked').val()\n" +
                    "                displayOptions.sortRules = $('input[name=sr]:checked').val()\n" +
                    "                displayOptions.details = $('#details').prop('checked')\n" +
                    "                numOfLabels = document.getElementById('numOfLabels')\n" +
                    "                displayOptions.numOfLabels = numOfLabels.options[numOfLabels.selectedIndex].value\n" +
                    "                return displayOptions\n" +
                    "            }\n" +
                    "\n" +
                    "            function sortByAbsScoreDescending(labels) {\n" +
                    "                labels.forEach(function(lb) {\n" +
                    "                    lb.rules = lb.rules.sort(function(a, b) {\n" +
                    "                        return Math.abs(b.score) - Math.abs(a.score)\n" +
                    "                    })\n" +
                    "                })\n" +
                    "            }\n" +
                    "\n" +
                    "            function sortByScoreAscending(labels) {\n" +
                    "                labels.forEach(function(lb) {\n" +
                    "                    lb.rules = lb.rules.sort(function(a, b) {\n" +
                    "                        return a.score - b.score\n" +
                    "                    })\n" +
                    "                })\n" +
                    "            }\n" +
                    "\n" +
                    "            function sortByScoreDescending(labels) {\n" +
                    "                labels.forEach(function(lb) {\n" +
                    "                    lb.rules = lb.rules.sort(function(a, b) {\n" +
                    "                        return b.score - a.score\n" +
                    "                    })\n" +
                    "                })\n" +
                    "            }\n" +
                    "\n" +
                    "            function indexOfLabels(labels, label) {\n" +
                    "                for (var i = 0; i < labels.length; i++) {\n" +
                    "                    lb = labels[i]\n" +
                    "                    for (var k in lb) {\n" +
                    "                        if (lb[k] == label) {\n" +
                    "                            return true\n" +
                    "                        }\n" +
                    "                    }\n" +
                    "                }\n" +
                    "\n" +
                    "                return false\n" +
                    "            }\n" +
                    "\n" +
                    "            function sortByAnti(labels, internalLabels) {\n" +
                    "                labels.forEach(function(lb) {\n" +
                    "                    if (indexOfLabels(internalLabels, lb.name) == true) {\n" +
                    "                        lb.rules = lb.rules.sort(function(a, b) {\n" +
                    "                            return a.score - b.score\n" +
                    "                        })\n" +
                    "                    } else {\n" +
                    "                        lb.rules = lb.rules.sort(function(a, b) {\n" +
                    "                            return b.score - a.score\n" +
                    "                        })\n" +
                    "                    }\n" +
                    "                })\n" +
                    "            }\n" +
                    "\n" +
                    "            function sortByViewOptions(data, displayOptions) {\n" +
                    "                if (displayOptions.sortRules === 'abs') {\n" +
                    "                    data.forEach(function (row) {\n" +
                    "                        sortByAbsScoreDescending(row.TP)\n" +
                    "                        sortByAbsScoreDescending(row.FP)\n" +
                    "                        sortByAbsScoreDescending(row.FN)\n" +
                    "                        sortByAbsScoreDescending(row.TN)\n" +
                    "                    })\n" +
                    "                } else if (displayOptions.sortRules === 'ascending') {\n" +
                    "                    data.forEach(function (row) {\n" +
                    "                        sortByScoreAscending(row.TP)\n" +
                    "                        sortByScoreAscending(row.FP)\n" +
                    "                        sortByScoreAscending(row.FN)\n" +
                    "                        sortByScoreAscending(row.TN)\n" +
                    "                    })\n" +
                    "                } else if (displayOptions.sortRules == 'descending') {\n" +
                    "                    data.forEach(function (row) {\n" +
                    "                        sortByScoreDescending(row.TP)\n" +
                    "                        sortByScoreDescending(row.FP)\n" +
                    "                        sortByScoreDescending(row.FN)\n" +
                    "                        sortByScoreDescending(row.TN)\n" +
                    "                    })\n" +
                    "                } else if (displayOptions.sortRules == 'anti') {\n" +
                    "                    data.forEach(function (row) {\n" +
                    "                        sortByAnti(row.TP, row.idlabels.internalLabels)\n" +
                    "                        sortByAnti(row.FP, row.idlabels.internalLabels)\n" +
                    "                        sortByAnti(row.FN, row.idlabels.internalLabels)\n" +
                    "                        sortByAnti(row.TN, row.idlabels.internalLabels)\n" +
                    "                    })\n" +
                    "                } else {\n" +
                    "                    alert(displayOptions.sortRules)\n" +
                    "                }\n" +
                    "\n" +
                    "                // Sort data by displayOptions\n" +
                    "                if (displayOptions.sortDocs == 'confide') {\n" +
                    "                    data.sort(function (a, b) {\n" +
                    "                        return a.idlabels.overlap - b.idlabels.overlap\n" +
                    "                    })\n" +
                    "                } else if (displayOptions.sortDocs == 'mistake') {\n" +
                    "                    data.sort(function (a, b) {\n" +
                    "                        return b.idlabels.overlap - a.idlabels.overlap\n" +
                    "                    })\n" +
                    "                } else {\n" +
                    "                }\n" +
                    "                return data\n" +
                    "            }\n" +
                    "\n" +
                    "            $(document).ready(function () {\n" +
                    "                var displayOptions = viewOptions()\n" +
                    "                render(sortByViewOptions(dataFromJson(), displayOptions), displayOptions) \n" +
                    "\n" +
                    "                $('#btn-submit').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#TP').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#FP').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#FN').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#TN').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#Confide').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#Mistake').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#abs').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#ascending').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#descending').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#anti').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#details').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "                $('#test').click(function () {\n" +
                    "                    refresh()\n" +
                    "                })\n" +
                    "            })\n" +
                    "\n" +
                    "            function makeTextFile(text) {\n" +
                    "                var data = new Blob([text], {type: 'text/plain'});\n" +
                    "\n" +
                    "                textFile = window.URL.createObjectURL(data);\n" +
                    "\n" +
                    "                return textFile;\n" +
                    "            }\n" +
                    "\n" +
                    "            function refreshTable(displayOptions) {\n" +
                    "                colums = ['TP', 'FP', 'FN', 'TN']\n" +
                    "                feedbackColumns = ['failure', 'incomplete', 'bad_rule', 'bad_matching']\n" +
                    "                selections = {'failure':0, 'incomplete':0, 'bad_rule':0, 'bad_matching':0}\n" +
                    "                startIndex = 2\n" +
                    "\n" +
                    "                var table = document.getElementById(\"mytable\");\n" +
                    "                var rows = table.getElementsByTagName(\"tr\");\n" +
                    "                for (var row = 0; row < rows.length; row++) {\n" +
                    "                    var cols = rows[row].children;\n" +
                    "                    var selId = 'sel' + row\n" +
                    "\n" +
                    "                    sel = document.getElementById(selId)\n" +
                    "                    if (sel != undefined) {\n" +
                    "                        selections[sel.options[sel.selectedIndex].value] += 1\n" +
                    "                    }\n" +
                    "\n" +
                    "                    for (var i = 0; i < colums.length; i++) {\n" +
                    "                        if (displayOptions.fields.indexOf(colums[i]) == -1) {\n" +
                    "                            cols[startIndex + i].style.display = 'none';\n" +
                    "                        } else {\n" +
                    "                            cols[startIndex + i].style.display = 'table-cell';\n" +
                    "                        }\n" +
                    "                    }\n" +
                    "                }\n" +
                    "                \n" +
                    "                var table = document.getElementById(\"data-feedbackTable\")\n" +
                    "                var row = table.getElementsByTagName(\"tr\");\n" +
                    "                cols = row[0].children\n" +
                    "                for (i = 0; i < cols.length; i++) {\n" +
                    "                    cols[i].innerHTML = selections[feedbackColumns[i]]\n" +
                    "                }\n" +
                    "\n" +
                    "            }\n" +
                    "\n" +
                    "            function dataFromJson() {\n" +
                    "                return JSON.parse($('#raw-data').html())\n" +
                    "            }\n" +
                    "        </script>\n" +
                    "\n" +
                    "        <style>\n" +
                    "            #feedbackTable{\n" +
                    "                border: 1px solid black;\n" +
                    "                border-collapse: collapse;\n" +
                    "            }\n" +
                    "            #mytable{\n" +
                    "                border: 1px solid black;\n" +
                    "                border-collapse: collapse;\n" +
                    "                word-wrap: break-word;\n" +
                    "                table-layout:fixed;\n" +
                    "            }\n" +
                    "            #optionTable {\n" +
                    "                border : 1px solid black;\n" +
                    "                border-collapse: collapse;\n" +
                    "                float : center;\n" +
                    "                width : 40%;\n" +
                    "            }\n" +
                    "            .ruleList {\n" +
                    "                list-style-type: square;\n" +
                    "            }\n" +
                    "            .mytext {\n" +
                    "                width: 100%;\n" +
                    "                height:200px;\n" +
                    "            }\n" +
                    "            strong{\n" +
                    "                font-weight:bold;\n" +
                    "            }\n" +
                    "        </style>\n" +
                    "<script id=\"raw-data\" type=\"application/json\">\n";
    public static final String END_HTML =
            "\n" +
                    "</script>\n" +
                    "</body></html>\n";

    // public static final String CLUSTER_NAME = "ohsumed_20000"; // picked up from .properties file
    public static final String DOCUMENT_TYPE = "document";



}