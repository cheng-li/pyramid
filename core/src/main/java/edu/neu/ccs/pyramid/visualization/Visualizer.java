package edu.neu.ccs.pyramid.visualization;

import com.google.gson.*;
import edu.neu.ccs.pyramid.util.DirWalker;
import org.apache.http.HttpEntity;
import org.apache.http.HttpHost;
import org.apache.http.entity.ContentType;
import org.apache.http.nio.entity.NStringEntity;
import org.apache.http.util.EntityUtils;
import org.elasticsearch.client.Response;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestClientBuilder;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Created by shikhar on 6/28/17.
 */
public class Visualizer implements AutoCloseable{
    private JsonParser jsonParser = null;
    private Gson gson = null;
    private JsonArray writeRulePositions = null;
    private String writeRuleField = null;
    private Logger logger;
    private RestClient esClient;

    public Visualizer(Logger logger, List<String> hosts, List<Integer> ports) {
        jsonParser = new JsonParser();
        gson = new GsonBuilder().serializeNulls().create();
        this.logger = logger;

        HttpHost[] httpHosts = new HttpHost[hosts.size()];
        for (int i=0;i< hosts.size();i++){
            HttpHost host = new HttpHost(hosts.get(i), ports.get(i),"http");
            httpHosts[i] = host;
        }
        esClient = RestClient.builder(httpHosts).build();
    }

    public void close(){
        try {
            esClient.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void produceHtml(File inputPath){
        File inputDir = getInputDir(inputPath.getAbsolutePath());
        processFolder(inputDir);
    }


    public void processFolder(File inputDir) {
        /*   read config files   */
        String configString = null;
        String dataString = null;
        try {
            configString =
                    Files.lines(Paths.get(inputDir.getAbsolutePath(), Properties.CONFIG_FILE_NAME))
                            .collect(Collectors.joining());
            dataString =
                    Files.lines(Paths.get(inputDir.getAbsolutePath(), Properties.DATA_FILE_NAME))
                            .collect(Collectors.joining());
        } catch (IOException e) {
            e.printStackTrace();
            logger.log(Level.SEVERE,"Exception in reading one of the config files in json. Exiting ...");
            System.exit(1);
        }
        JsonObject config = jsonParser.parse(configString).getAsJsonObject();
        JsonObject data = jsonParser.parse(dataString).getAsJsonObject();

        String fields = config.get("train.feature.ngram.extractionFields").getAsString();
        String esIndex = config.get("index.indexName").getAsString();
        String classNumber = data.get("numClassesInModel").getAsString();


        // run a loop for all the json files. Call processFile() for every file
        List<File> allFiles= null;
        try {
            allFiles = DirWalker.getFiles(inputDir.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
        for (File jsonReport : allFiles) {

            /*  Ignoring all files other than "report*.json"  */
            String fileName = jsonReport.getName();
            if ( ! fileName.endsWith(".json"))
                continue;
            if ( ! fileName.startsWith("report"))
                continue;
            logger.info("processing file "+jsonReport.getName());
            processFile(jsonReport, esIndex, fields);
        }
    }

    private void processFile(File reportFile, String esIndex, String fields) {
        String inputFileName = reportFile.getName();
        String outputFileName = inputFileName.replaceAll(".json", ".html");
        File outputFile = new File(reportFile.getParent(), outputFileName);
        String jsonString = null;
        try {
            jsonString = new String(Files.readAllBytes(Paths.get(reportFile.toURI())));
        } catch (IOException e) {
            e.printStackTrace();
        }
        String htmlTable = jsonToHtmlTable(jsonString, esIndex, fields);

        String htmlReport = Properties.PRE_HTML + htmlTable + Properties.END_HTML;
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputFile.getAbsolutePath()))){
            writer.write(htmlReport);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /*   corresponds to createTable() of visualizer.py   */
    private String jsonToHtmlTable(String jsonString,String esIndex, String fields) {
        JsonElement json_file = (new JsonParser()).parse(jsonString);
        JsonArray jsonRows = json_file.getAsJsonArray();
        int lineCount = 0;
        JsonArray output = new JsonArray();
        for (JsonElement jsonRow_je : jsonRows) {
            JsonObject jsonRow = jsonRow_je.getAsJsonObject();
            lineCount += 1;
            output.add(createRow(jsonRow, esIndex, fields, lineCount));

        }
        return gson.toJson(output);
    }

    /*   corresponds to each iteration of the loop in createTable() of viusualizer.py   */
    private JsonObject createRow(JsonObject jsonRow, String esIndex, String fields, int lineCount) {
        JsonObject thisRow = new JsonObject();
        Map<String, JsonElement> labelsMap = new HashMap<>();
        labelsMap.put("id", jsonRow.get("id"));
        labelsMap.put("internalId", jsonRow.get("internalId"));
        labelsMap.put("internalLabels", jsonRow.get("internalLabels"));
        JsonParser jp = new JsonParser();
        labelsMap.put("feedbackSelect", jp.parse("none"));
        labelsMap.put("feedbackText", jp.parse(""));// WARNING: "" is parsed to null in json


        /*  Internal Labels:  */
        Map<String , JsonElement> internalLabels = new HashMap<>();
        JsonArray labels_jsonArr = jsonRow.get("labels").getAsJsonArray();
        JsonArray internalLabels_jsonArr = jsonRow.get("internalLabels").getAsJsonArray();
        int totalLabels = labels_jsonArr.size();
        labelsMap.put("internalLabels", new JsonArray());
        for (int i=0; i<totalLabels; i++) {
            internalLabels.put(internalLabels_jsonArr.get(i).getAsString(), labels_jsonArr.get(i));
            JsonObject jo = new JsonObject();
            jo.add(internalLabels_jsonArr.get(i).getAsString(), labels_jsonArr.get(i));
            labelsMap.get("internalLabels").getAsJsonArray().add(jo);
        }


        /*  Predictions:  */
        Map<String, JsonElement> predictions = new HashMap<>();
        JsonArray predictions_jsons = jsonRow.get("prediction").getAsJsonArray();
        JsonArray internalPredictions_jsons = jsonRow.get("internalPrediction").getAsJsonArray();
        int totalPredictions = predictions_jsons.size();
        labelsMap.put("predictions", new JsonArray());
        for (int i = 0; i < totalPredictions; i++) {
            // TODO: use getAsString() instead of toString() ?
            predictions.put(internalPredictions_jsons.get(i).getAsString(), predictions_jsons.get(i));
            JsonObject jo = new JsonObject();
            jo.add(internalPredictions_jsons.get(i).getAsString(), predictions_jsons.get(i));
            JsonArray ja= labelsMap.get("predictions").getAsJsonArray();
            ja.add(jo);
        }


        /*  overlap, recall and precision  */
        Set<String> internalLabelIds = new HashSet<>(internalLabels.keySet());
        Set<String> predictionsIds = new HashSet<>(predictions.keySet());
        Set<String> intersection = new HashSet<>(internalLabelIds);
        intersection.retainAll(predictionsIds);
        Set<String> union = new HashSet<>(internalLabelIds);
        union.addAll(predictionsIds);
        if (union.size() == 0)
            labelsMap.put("overlap", gson.toJsonTree("N/A"));
        else
            labelsMap.put("overlap", gson.toJsonTree(String.format("%.2f", (double) intersection.size() / union.size())));
        if (internalLabelIds.size() == 0)
            labelsMap.put("recall", gson.toJsonTree("N/A"));
        else
            labelsMap.put("recall", gson.toJsonTree(String.format("%.2f", (double) intersection.size() / internalLabelIds.size())));
        if (predictionsIds.size() == 0)
            labelsMap.put("precision", gson.toJsonTree("N/A"));
        else
            labelsMap.put("precision", gson.toJsonTree(String.format("%.2f", (double) intersection.size() / predictionsIds.size())));



        /*  probForPredictedLabels  */
        thisRow.add("probForPredictedLabels", jsonRow.get("probForPredictedLabels"));



        /*  column 2; predicted ranking  */
        JsonArray predictedRankings_ja = jsonRow.get("predictedRanking").getAsJsonArray();
        PredictedRanking[] predictedRankings = gson.fromJson(predictedRankings_ja, PredictedRanking[].class);
        PredictedRanking[] updatedPredictedRankings = new PredictedRanking[predictedRankings.length];
        List<Integer> r = new ArrayList<>(predictedRankings.length);
        JsonElement predictedRanking_json = null;
        for (int i=0; i<predictedRankings.length; i++) {
            PredictedRanking predictedRanking = predictedRankings[i]; // ~ label in visualizer.py
            predictedRanking_json = gson.toJsonTree(predictedRanking);
            JsonElement classIndex_je = predictedRanking_json.getAsJsonObject().get("classIndex");
            if (internalLabels_jsonArr.contains(classIndex_je) && internalPredictions_jsons.contains(classIndex_je)) {
                predictedRanking_json.getAsJsonObject().addProperty("type", "TP");
            } else if (! internalLabels_jsonArr.contains(classIndex_je) && internalPredictions_jsons.contains(classIndex_je))
                predictedRanking_json.getAsJsonObject().addProperty("type","FP");
            else if (internalLabels_jsonArr.contains(classIndex_je) && ! internalPredictions_jsons.contains(classIndex_je))
                predictedRanking_json.getAsJsonObject().addProperty("type", "FN");
            else
                predictedRanking_json.getAsJsonObject().addProperty("type", "");
            PredictedRanking newPredictedRanking = gson.fromJson(predictedRanking_json, PredictedRanking.class);
            updatedPredictedRankings[i] = newPredictedRanking;

            // updating r for some-more-labelsMap
            r.add(includesLabel(predictedRanking.className, internalLabels));
        }
        JsonArray updatedPredictedRankings_json = gson.toJsonTree(updatedPredictedRankings).getAsJsonArray();
        thisRow.add("predictedRanking", updatedPredictedRankings_json);



        /*   predictedLabelSetRankings   */
        JsonArray predictedLabelSetRankings = jsonRow.get("predictedLabelSetRanking").getAsJsonArray();

        // making a Set copy of these so that contains() is constant time:
        Set<Integer> internalLabels_ints = jsonArrToIntSet(internalLabels_jsonArr);
        Set<Integer> internalPredictions_ints = jsonArrToIntSet(internalPredictions_jsons);

        for (JsonElement labelsJElement : predictedLabelSetRankings) {
            JsonObject labels = labelsJElement.getAsJsonObject();
            // labels ~ labels in visualizer.py
            JsonArray predictedInternalLabels = labels.get("internalLabels").getAsJsonArray();
            labels.add("types", new JsonArray());
            for (JsonElement index_je : predictedInternalLabels) {
                int index = index_je.getAsInt();
                if (internalLabels_ints.contains(index) && internalPredictions_ints.contains(index))
                    labels.get("types").getAsJsonArray().add("TP");
                else if (!internalLabels_ints.contains(index) && internalPredictions_ints.contains(index))
                    labels.get("types").getAsJsonArray().add("FP");
                else if (internalLabels_ints.contains(index) && !internalPredictions_ints.contains(index))
                    labels.get("types").getAsJsonArray().add("FN");
                else
                    labels.get("types").getAsJsonArray().add("");
            }
        }
        thisRow.add("predictedLabelSetRanking", jsonRow.get("predictedLabelSetRanking"));



        /*   some-more-labelsMap   */
        double sumOfR = 0;
        double sumOfPrec = 0;
        double prec = 0;
        int last = 0;
        for (int i=0; i<r.size(); i++) {
            if (r.get(i) == 1) {
                sumOfR += 1;
                prec = sumOfR / (i + 1);
                sumOfPrec += prec;
                last = i+1;
            }
        }
        int intLblsSize = internalLabelIds.size();
        if (intLblsSize == 0)
            labelsMap.put("ap", gson.toJsonTree("N/A"));
        else
            labelsMap.put("ap", gson.toJsonTree(String.format("%.2f", (sumOfPrec / intLblsSize))));
        if (sumOfR < intLblsSize)
            labelsMap.put("rankoffullrecall", gson.toJsonTree("N/A"));
        else
            labelsMap.put("rankoffullrecall", gson.toJsonTree(last)); // warning: storing integer as string. might create problems later

        thisRow.add("idlabels", gson.toJsonTree(labelsMap));


        /*   column 3 : ES   */
        Response response = null;
        String jsonResponse = null;
        try {
            response = esClient.performRequest(
                    "GET",
                    esIndex + "/" + Properties.DOCUMENT_TYPE + "/" + URLEncoder.encode(labelsMap.get("id").getAsString(),"UTF-8"),
                    Collections.emptyMap()
            );
            jsonResponse = EntityUtils.toString(response.getEntity());
        } catch (IOException e) {
            e.printStackTrace();
        }

        JsonObject docSource = jsonParser.parse(jsonResponse).getAsJsonObject()
                .get("_source").getAsJsonObject();

        thisRow.add("text", new JsonObject());
        thisRow.add("others", new JsonObject());
        for (Map.Entry<String, JsonElement> entry : docSource.entrySet()) {
            String k = entry.getKey();
            JsonElement v = entry.getValue();
            if (entry.getKey().equals(fields)) {
                thisRow.get("text").getAsJsonObject().add(k,v);
            } else
                thisRow.get("others").getAsJsonObject().add(k,v);
        }


        /*   column 4 - 7 TP FP FN TN   */
        createTFPNColumns(esIndex, jsonRow, lineCount, thisRow);

        return thisRow;
    }

    private void createTFPNColumns(String esIndex, JsonObject jsonRow, int lineCount, JsonObject thisRow) {
        String id = jsonRow.get("id").getAsString();
        JsonArray classScoreCalculations = jsonRow.get("classScoreCalculations").getAsJsonArray();
        JsonArray internalLabels = jsonRow.get("internalLabels").getAsJsonArray();
        JsonArray internalPrediction = jsonRow.get("internalPrediction").getAsJsonArray();

        thisRow.add("TP", new JsonArray());

        for (JsonElement eachClass : classScoreCalculations) {
            JsonElement internalClassIndex = eachClass.getAsJsonObject().get("internalClassIndex");
            /*   column 4 TP   */
            if (internalLabels.contains(internalClassIndex) && internalPrediction.contains(internalClassIndex)) {
                thisRow.get("TP").getAsJsonArray().add(writeClass(esIndex, id, lineCount, eachClass.getAsJsonObject()));
            }
        }

        // TODO: combine the following 3 for loops into one if the output does not change


        /*   column 5 FP   */
        thisRow.add("FP", new JsonArray());
        for (JsonElement eachClass : classScoreCalculations){
            JsonElement internalClassIndex = eachClass.getAsJsonObject().get("internalClassIndex");
            if (!internalLabels.contains(internalClassIndex) && internalPrediction.contains(internalClassIndex)) {
                thisRow.get("FP").getAsJsonArray().add(writeClass(esIndex, id, lineCount, eachClass.getAsJsonObject()));
            }
        }

        /*   column 6 FN   */
        thisRow.add("FN", new JsonArray());
        for (JsonElement eachClass : classScoreCalculations) {
            JsonElement internalClassIndex = eachClass.getAsJsonObject().get("internalClassIndex");
            if (internalLabels.contains(internalClassIndex) && !internalPrediction.contains(internalClassIndex)) {
                thisRow.get("FN").getAsJsonArray().add(writeClass(esIndex, id, lineCount, eachClass.getAsJsonObject()));
            }
        }

        /*   column 7 TN   */
        thisRow.add("TN", new JsonArray());
        for (JsonElement eachClass : classScoreCalculations) {
            JsonElement internalClassIndex = eachClass.getAsJsonObject().get("internalClassIndex");
            if (!internalLabels.contains(internalClassIndex) && !internalPrediction.contains(internalClassIndex)) {
                thisRow.get("TN").getAsJsonArray().add(writeClass(esIndex, id, lineCount, eachClass.getAsJsonObject()));
            }
        }
    }

    private JsonObject writeClass(String esIndex, String id, int lineCount, JsonObject eachClass) {
        JsonObject thisClass = new JsonObject();
        thisClass.add("id", eachClass.get("internalClassIndex"));
        String name = eachClass.get("className").getAsString();

        /*
        * the "if" block is never executed as classDescription is hardcoded to be {} in visualizer.py.
        * So, directly coding the else block
        */
        name += " : MISSING DESCRIPTION";

        thisClass.add("name", gson.toJsonTree(name));
        thisClass.add("classProbability", eachClass.get("classProbability"));
        thisClass.add("totalScore", eachClass.get("classScore"));

        // default rule number is 6  (comment copied from visualizer.py)
        int start = 0;
        JsonArray rules = eachClass.get("rules").getAsJsonArray();
        JsonObject rule0 = rules.get(0).getAsJsonObject();
        if (! rule0.keySet().contains("checks")) {
            thisClass.add("prior", rule0.get("score"));
            start = 1;
        }

        JsonArray allPos = new JsonArray();
        thisClass.add("rules", new JsonArray());
        String field = "";
        for (int i = start; i < rules.size(); i++) {
            JsonObject rulei = rules.get(i).getAsJsonObject();
            JsonObject oneRule = writeRule(esIndex, id, lineCount, i, rulei);
            JsonArray pos = writeRulePositions;
            field = writeRuleField;
            thisClass.get("rules").getAsJsonArray().add(oneRule);
            allPos.addAll(pos);
        }
        thisClass.addProperty("allPos", allPos.toString());//visualizer.py converts this to a string but it should actually be left as an array
        thisClass.addProperty("field", field);

        return thisClass;
    }

    private JsonObject writeRule(String esIndex, String id, int lineCount, int i, JsonObject rulei) {
        JsonObject oneRule = new JsonObject();

        oneRule.add("score", rulei.get("score").getAsJsonPrimitive());
        oneRule.add("checks", new JsonArray());


        // BW: add all position into one list and field
        JsonArray allPos = new JsonArray();
        String field = "";
        JsonArray checks = rulei.get("checks").getAsJsonArray();
        for (JsonElement check_je : checks) {
            JsonObject check = check_je.getAsJsonObject();
            JsonObject checkOneRule = new JsonObject();
            JsonArray pos = new JsonArray();
            JsonObject feature = check.get("feature").getAsJsonObject();

            double featureValue = check.get("feature value").getAsDouble();
            Set<String> featureKeys = check.get("feature").getAsJsonObject().keySet();
            if (featureValue != 0.0 && featureKeys.contains("ngram")) {
                pos = getPositions(
                        esIndex,
                        id,
                        feature.get("field").getAsString(),
                        feature.get("ngram"),
                        feature.get("slop").getAsInt(),
                        feature.get("inOrder").getAsBoolean()
                );
            }
            Set<String> checkFeatures = feature.keySet();
            if (!checkFeatures.contains("ngram")) {
                checkOneRule.add("name", feature.get("name"));
            } else {
                checkOneRule.add("ngram", feature.get("ngram"));
                checkOneRule.add("field", feature.get("field"));
                checkOneRule.add("slop", feature.get("slop"));
                field = checkOneRule.get("field").getAsString();
            }
            checkOneRule.add("value", check.get("feature value"));
            checkOneRule.add("relation", check.get("relation"));
            checkOneRule.add("threshold", check.get("threshold"));
            checkOneRule.addProperty("highlights", pos.toString()); // converting 'pos' to string, like in visualizer.py
            allPos.addAll(pos);
            oneRule.get("checks").getAsJsonArray().add(checkOneRule);

        }
        writeRulePositions = allPos;
        writeRuleField = field;
        return oneRule;
    }

    private JsonArray getPositions(String esIndex, String id, String field, JsonElement keywords, int slop, boolean inOrder) {
        // TODO: Debug
//        System.out.println("ID: " + id);
//        System.out.println("Words: " + keywords);
//        System.out.println("slope: " + slop);

//        System.out.println("IN");
        JsonArray clauses = new JsonArray();
        String[] keywordsArr = keywords.getAsString().split("\\s");

        String queryJson = null;
        if (keywordsArr.length == 1) {
            queryJson = createUnigramQueryJson(id, field, keywordsArr[0]);
        } else {
            for (String keyword : keywordsArr) {
                JsonObject field_j = new JsonObject();
                field_j.add(field, gson.toJsonTree(keyword));
                JsonObject clause = new JsonObject();
                clause.add("span_term", field_j);
                clauses.add(clause);
            }
            //  added to handle lucene changes that require a min. of 2 span fields
//            System.out.println("clauses_original: " + clauses);
//            if (clauses.size() == 1)
//                clauses.add(clauses.get(0));
//            System.out.println("clauses_new: " + clauses);
            queryJson = createQueryJson(id, inOrder, slop, field, clauses);
        }

//        System.out.println(queryJson);
//        try {
//            System.in.read();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        HttpEntity httpEntity = new NStringEntity(queryJson, ContentType.APPLICATION_JSON);
        String responseStr = null;
        try {
            Response response =
                    esClient.performRequest(
                            "GET",
                            esIndex + "/" + Properties.DOCUMENT_TYPE + "/" + "_search",
                            Collections.emptyMap(),
                            httpEntity
                    );
            responseStr = EntityUtils.toString(response.getEntity());
        } catch (IOException e) {
            e.printStackTrace();
        }

        JsonObject responseObj = jsonParser.parse(responseStr).getAsJsonObject();
        JsonArray hits = responseObj.get("hits").getAsJsonObject().get("hits").getAsJsonArray();

        if (hits.size() == 0) {
            return new JsonArray();
        }

        JsonArray positions = new JsonArray(); // positions is an array of arryas. not sure if inefficient. TODO: change later if inefficient
        JsonObject hit0 = hits.get(0).getAsJsonObject();
        Set<String> fields = hit0.get("highlight").getAsJsonObject().keySet();
        for (String higlightField : fields) {
            String text = hit0.get("_source").getAsJsonObject().get(higlightField).getAsString();
            JsonArray highlights = hit0.get("highlight").getAsJsonObject().get(higlightField).getAsJsonArray();
            for (JsonElement hl_je : highlights) {
                String hl = hl_je.getAsString();
//                System.out.println("hl" + hl);
                String cleanHl = hl.replaceAll("<\\/?em>", ""); // remove <em> and </em>
                int baseIndex = text.indexOf(cleanHl);

                if (baseIndex == -1)
                    continue;

                while (hl.indexOf("<em>") != -1) {
                    int start = hl.indexOf("<em>") + baseIndex;
                    hl = hl.replaceFirst("<em>", "");
                    int end = hl.indexOf("</em>") + baseIndex;
                    hl = hl.replaceFirst("</em>", "");

                    JsonArray curPos = gson.toJsonTree(new int[]{start, end}).getAsJsonArray();
//                    System.out.println("curPos:" + curPos);
//                    try {
//                        System.in.read();
//                    } catch (IOException e) {
//                        e.printStackTrace();
//                    }
                    positions.add(curPos); // positions is a array of arrays. (each curPos is an array)
                }
            }
        }

//        //TODO: Debug part
//        System.out.println(positions);
//        try {
//            System.in.read();
//        } catch (IOException e) {
//
//
//        }
//
//        //TODO: End of debugging part

        return positions;
    }

    private String createUnigramQueryJson(String id , String field, String keyword) {
        return
                "{\n" +
                        "  \"query\": {\n" +
                        "    \"bool\": {\n" +
                        "      \"filter\": {\n" +
                        "        \"ids\": {\n" +
                        "          \"values\": [\n" +
                        "            " + gson.toJsonTree(id) + "\n" +
                        "          ]\n" +
                        "        }\n" +
                        "      },\n" +
                        "      \"must\": {\n" +
                        "        \"span_term\": {\n" +
                        "          \"" + field + "\": " + "\"" + keyword + "\"" + "\n" +
                        "        }\n" +
                        "      }\n" +
                        "    }\n" +
                        "  },\n" +
                        "  \"explain\": false,\n" +
                        "  \"size\": 1,\n" +
                        "  \"highlight\": {\n" +
                        "    \"fields\": {\n" +
                        "      " + gson.toJsonTree(field) + ": {}\n" +
                        "    }\n" +
                        "  }\n" +
                        "}";

    }
    private String createQueryJson(String id , boolean inOrder, int slop, String field, JsonArray clauses) {
        return
                "{\n" +
                        "  \"query\": {\n" +
                        "    \"bool\": {\n" +
                        "      \"filter\": {\n" +
                        "        \"ids\": {\n" +
                        "          \"values\": [\n" +
                        "            " + gson.toJsonTree(id) + "\n" +
                        "          ]\n" +
                        "        }\n" +
                        "      },\n" +
                        "      \"must\": {\n" +
                        "        \"span_near\": {\n" +
                        "          \"in_order\": " + gson.toJsonTree(inOrder) + ",\n" +
                        "          \"clauses\": " + clauses + ",\n" +
                        "          \"slop\": " + gson.toJsonTree(slop) + ",\n" +
                        "          \"collect_payloads\": false\n" +
                        "        }\n" +
                        "      }\n" +
                        "    }\n" +
                        "  },\n" +
                        "  \"explain\": false,\n" +
                        "  \"size\": 1,\n" +
                        "  \"highlight\": {\n" +
                        "    \"fields\": {\n" +
                        "      " + gson.toJsonTree(field) + ": {}\n" +
                        "    }\n" +
                        "  }\n" +
                        "}";

    }

    private int includesLabel(String className, Map<String, JsonElement> internalLabels) {
        for (Map.Entry<String, JsonElement> internalLabel : internalLabels.entrySet()) {
            String label = internalLabel.getValue().getAsString();
            if (className.equals(label))
                return 1;
        }
        return 0;
    }

    private static Set<Integer> jsonArrToIntSet(JsonArray jsonArray) {
        HashSet<Integer> set = new HashSet<>();
        for (JsonElement je : jsonArray) {
            Integer i = je.getAsInt();
            set.add(i);
        }
        return set;
    }

    private static int[] jsonArrToIntArr(JsonArray jsonArray) {
        int size = jsonArray.size();
        int[] intArr = new int[size];
        for (int i = 0; i <size; i++) {
            JsonElement je = jsonArray.get(i);
            intArr[i] = je.getAsInt();
        }
        return intArr;
    }

    private static File getInputDir(String inputStr) {
        File inputFile = new File(inputStr);
        if (! inputFile.exists())
            throw new IllegalArgumentException("input location given does not exist. Please give th full path");

        if (inputFile.isDirectory()) {
            return inputFile;
        } else if (inputFile.isFile()) {
            return inputFile.getParentFile();
        } else
            throw new IllegalArgumentException("Input given is neither a File nor a directory");
    }

}