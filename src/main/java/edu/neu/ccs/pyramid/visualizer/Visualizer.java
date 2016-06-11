/**
 *******************************************************************************
 * Copyright by Bishwajeet Dey.
 * All rights reserved.
 *******************************************************************************/
package edu.neu.ccs.pyramid.visualizer;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.elasticsearch.action.get.GetRequestBuilder;
import org.elasticsearch.action.get.GetResponse;
import org.elasticsearch.common.base.Preconditions;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.Maps;

import edu.neu.ccs.pyramid.configuration.Config;

/**
 * Description
 * @author <a href="mailto:deyb@ccs.neu.edu">Bishwajeet Dey</a>
 *
 * @version 1.0.0
 */
public class Visualizer {
    
    private final VisualizerConfig config;

    public Visualizer(VisualizerConfig config) {
        Preconditions.checkNotNull(config);
        this.config = config;
    }

    public void visualize() throws IOException {
        final String outputFileName = "viewer";

        final String indPerformanceName = "individual_performance";
        final String topName            = "top_features";
        final String configName         = "data_config";
        final String dataName           = "data_info";
        final String modelName          = "model_config";
        final String performanceName  = "performance";
        
        final File inputTop           = new File(config.getInputFileBaseDir(), topName + ".json");
         File outputPath              = new File(config.getInputFileBaseDir(), topName + ".html");
        
        createTopFeatureHTML(inputTop, outputPath);
        
        // create individual performance html
        final File inputIndPer = new File(config.getInputFileBaseDir(), indPerformanceName + ".json");
        final File inputAllPer = new File(config.getInputFileBaseDir(), performanceName + ".json");
        outputPath =  new File(config.getInputFileBaseDir(), indPerformanceName + ".html");
        createIndPerHTML(inputIndPer, inputAllPer, outputPath);
        
        File inputData   = new File(config.getInputFileBaseDir(), dataName + ".json");
        File inputModel  = new File(config.getInputFileBaseDir(), modelName + ".json");
        File inputConfig = new File(config.getInputFileBaseDir(), configName + ".json");
        outputPath       = new File(config.getInputFileBaseDir(),  "metadata.html");
        createMetaDataHTML(inputData, inputModel, inputConfig, outputPath);
        final Map<String, String> classDescription = new HashMap<>();
        for (String line : FileUtils.readLines(config.getClassFile(), "UTF-8")) {
            line = line.trim();
            String[] lineInfo = line.split("\\t");
            if (lineInfo.length > 1) {
                final String className = lineInfo[0];
                final String classDesc = lineInfo[1];
                classDescription.put(className, classDesc);
            }
        }
        
        final List<String> skipJsonFiles = Arrays.asList(configName+".json", dataName + ".json", modelName + ".json", topName + ".json",
                         performanceName + ".json", indPerformanceName + ".json");
        
        if (config.getInputFile().isFile()) {
            parse(config.getInputFile(), 
                    new File(config.getInputFileBaseDir(), outputFileName 
                            + "_" 
                            + config.getInputFile().getName().substring(0, config.getInputFile().getName().lastIndexOf('.')) 
                            + ".html"), 
                    config.getNgramFields(),  classDescription);
        } else {
            for (File f : config.getInputFileBaseDir().listFiles()) {
                final String fileName = f.getName();
                if (!skipJsonFiles.contains(fileName)) {
                    outputPath = new File(config.getInputFileBaseDir(), 
                            outputFileName
                            + "_"
                            + f.getName().substring(0, f.getName().lastIndexOf('.'))
                            + ".html");
                    parse(f, outputPath, config.getNgramFields(), classDescription);
                }
            }
        }
        
    }
    
    private void parse(File input_json_file, File outputFileName, String fields, Map<String, String> classDescription) throws IOException {
        final List<Object> outputData = createTable(loadJson(input_json_file, new TypeReference<List<Object>>() {
        }), fields,  classDescription);
        final String output = getResourcesFileAsString("pre-data.html") + getJsonString(outputData) + getResourcesFileAsString("post-data.txt");
        
        writeStringToFile(outputFileName, output);
    }

    private List<Object> createTable(List<Object> data, String fields, Map<String, String> classDescription) throws IOException {
        int line_count = 0;
        List<Object> output = new ArrayList<>();
        
        for (Object rowData : data) {
            final Map<String, Object> row = (Map<String, Object>) rowData;
            
            final Map<String, Object> oneRow = new HashMap<>();
            line_count += 1;
            
            final List<Object> r = new ArrayList<>();
            final Map<String, Object> idLabels = new HashMap<>();
            idLabels.put("id", row.get("id"));
            idLabels.put("internalId", row.get("internalId"));
            idLabels.put("internalLabels", row.get("internalLabels"));
            idLabels.put("feedbackSelect", "none");
            idLabels.put("feedbackText", "");
            
            final List<Object> internalLabels = new ArrayList<>();
            final Set<String> releLabels = new HashSet<>();
            
            for (int i = 0, rowSize = ((List<Object>)row.get("labels")).size(); i < rowSize; i++) {
                Map<Object, Object> label = new HashMap<>();
                label.put(((List<Object>)row.get("internalLabels")).get(i), ((List<Object>)row.get("labels")).get(i));
                internalLabels.add(label);
                releLabels.add(((List<String>)row.get("labels")).get(i));
            }
            
            final List<Object> predictions = new ArrayList<>();
            final Set<String> pres = new HashSet<>();
            
            for (int i = 0, rowSize = ((List<Object>)row.get("prediction")).size(); i < rowSize; i++) {
                Map<Object, Object> label = new HashMap<>();
                label.put(((List<Object>)row.get("internalPrediction")).get(i), ((List<Object>)row.get("prediction")).get(i));
                predictions.add(label);
                pres.add(((List<String>)row.get("prediction")).get(i));
            }
            
            idLabels.put("predictions", predictions);
            final Set<String> intersections = new HashSet<>(releLabels);
            intersections.retainAll(pres);
            
            final Set<String> unions = new HashSet<>(releLabels);
            unions.addAll(pres);
            
            if (unions.size() == 0) {
                idLabels.put("overlap", "N/A");
            } else {
                idLabels.put("overlap", String.format("%.2f", (double) intersections.size() / unions.size()));
            }
            
            if (releLabels.size() == 0) {
                idLabels.put("recall", "N/A");
            } else {
                idLabels.put("recall", String.format("%.2f", -(double) intersections.size() / releLabels.size()));
            }
            
            if (pres.size() == 0) {
                idLabels.put("precision", "N/A");
            } else {
                idLabels.put("precision", (double) intersections.size() / pres.size());
            }
            
            oneRow.put("probForPredictedLabels", row.get("probForPredictedLabels"));
            oneRow.put("predictedRanking", new ArrayList<>());
            
            for (Map<String, Object> label : (List<Map<String, Object>>)row.get("predictedRanking")) {
                final List<Object> curInternalPrediction = (List<Object>) row.get("internalPrediction");
                final List<Object> curInternalLabels = (List<Object>) row.get("internalLabels");
                
                final Object labelClassIndex = label.get("classIndex");
                
                if (curInternalLabels.contains(labelClassIndex) && curInternalPrediction.contains(labelClassIndex)) {
                    label.put("type", "TP");
                } else if (!curInternalLabels.contains(labelClassIndex) && curInternalPrediction.contains(labelClassIndex)) {
                    label.put("type", "FP");
                } else if (!curInternalLabels.contains(labelClassIndex) && !curInternalPrediction.contains(labelClassIndex)) {
                    label.put("type", "FN");
                } else {
                    label.put("type", "");
                }
                
                r.add(includesLabel(label.get("className"), internalLabels));
                ((List<Object>)oneRow.get("predictedRanking")).add(label);
            }
            
            for (Map<String, Object> labels : (List<Map<String, Object>>)row.get("predictedLabelSetRanking")) {
                labels.put("types", new ArrayList<>());
                
                final List<Object> curInternalLabels = (List<Object>) row.get("internalLabels");
                final List<Object> curInternalPrediction = (List<Object>) row.get("internalPrediction");
                
                for (Object index : (List<Object>) labels.get("internalLabels")) {
                    if (curInternalLabels.contains(index) && curInternalPrediction.contains(index)) {
                        ((List<Object>)labels.get("types")).add("TP");
                    } else if (!curInternalLabels.contains(index) && curInternalPrediction.contains(index)) {
                        ((List<Object>)labels.get("types")).add("FP");
                    } else if (curInternalLabels.contains(index) && !curInternalPrediction.contains(index)) {
                        ((List<Object>)labels.get("types")).add("FN");
                    } else {
                        ((List<Object>)labels.get("types")).add("");
                    }
                }
            }
            
            oneRow.put("predictedLabelSetRanking", row.get("predictedLabelSetRanking"));
            
            double sumOfR = 0.0;
            double sumOfPrec = 0.0;
            int last = 0;
            for (int i = 0; i < r.size(); i++) {
                if (r.get(i).equals(new Integer(1))) {
                    sumOfR += (Integer)r.get(i);
                    sumOfPrec += sumOfR / (i + 1);
                    last = i + 1;
                }
            }
            
            if (releLabels.size() == 0) {
                idLabels.put("ap", "N/A");
            } else {
                idLabels.put("ap", String.format("%.2f", sumOfPrec / releLabels.size()));
            }
            
            if (sumOfR < releLabels.size()) {
                idLabels.put("rankoffullrecall", "N/A");
            } else {
                idLabels.put("rankoffullrecall", last);
            }
            
            oneRow.put("idLabels", idLabels);
            
            GetResponse res = config.getClient().prepareGet(config.getEsIndexName(), "document", (String) row.get("id")).execute().actionGet();
            String keys = fields;
            oneRow.put("text",  new HashMap<String, Map<String, Object>>());
            oneRow.put("others", new HashMap<String, Map<String, Object>>());
            for (String key : ((Map<String, Object>)res.getContext().get("_source")).keySet()) {
                if (keys.equals(key)) {
                    ((Map<String, Object>) oneRow.get("text")).put(key, ((String)((Map<String, Object>)res.getContext().get("_source")).get(key)).replace("<", "&lt").replace(">", "&gt"));
                } else {
                    ((Map<String, Object>) oneRow.get("others")).put(key, ((String)((Map<String, Object>)res.getContext().get("_source")).get(key)));
                }
            }
                    
            createTFPNColumns(row, line_count, oneRow, classDescription);
            output.add(oneRow);
        }
        
        return output;
    }

    private void createTFPNColumns(Map<String, Object> row, int line_count, Map<String, Object> oneRow,
            Map<String, String> classDescription) throws IOException {
        final List<Object> tmpDict = new ArrayList<>();
        final Set<Object> labelSet = new HashSet<>();
        
        oneRow.put("TP", new ArrayList<>());
        
        for (Map<String, Object> clas : (List<Map<String, Object>>)row.get("classScoreCalculations")) {
            final List<Object> curInternalLabels = (List<Object>) row.get("internalLabels");
            final List<Object> curInternalPrediction = (List<Object>) row.get("internalPrediction");
            
            final Object internalClassIndex = clas.get("internalClassIndex");
            if (curInternalLabels.contains(internalClassIndex) && curInternalPrediction.contains(internalClassIndex)) {
                ((List<Object>)oneRow.get("TP")).add(writeClass(row.get("id"), line_count, clas, classDescription));
                oneRow.put("TP", oneRow.get("TP"));
            }
        }
        
    }

    private Object writeClass(Object docId, int line_count, Map<String, Object> clas,
            Map<String, String> classDescription) throws IOException {
        final Map<String, Object> oneClass = new HashMap<>();
        oneClass.put("id", clas.get("internalClassIndex"));
        
        String name = (String) clas.get("className");
        
        if (classDescription.containsKey(name)) {
            name = name + " : " + classDescription.get(name);
        } else {
            name = name + " : " + "MISSING DESCRIPTION";
        }
        
        oneClass.put("name", name);
        oneClass.put("classProbability", clas.get("classProbability"));
        oneClass.put("totalScore", clas.get("classScore"));
        
        int start = 0;
        if (!((List<Map<String, Object>>)clas.get("rules")).get(0).containsKey("checks")) {
            oneClass.put("prior", ((List<Map<String, Object>>)clas.get("rules")).get(0).get("score"));
            start = 1;
        }
        
        oneClass.put("rules", new ArrayList<>());
        List<Object> allPos = new ArrayList<>();
        String field = "";
        
        for (int i = start, size = ((List<Object>)clas.get("rules")).size(); i < size; i++) {
            final List<Object> writeRuleResult = writeRule(docId, line_count, i, ((List<Map<String, Object>>)clas.get("rules")).get(i));
            final Object oneRule = writeRuleResult.get(0);
            final Object pos = writeRuleResult.get(1);
            field = (String) writeRuleResult.get(2);
            
            ((List<Object>)oneClass.get("rules")).add(oneRule);
            oneClass.put("rules", oneClass.get("rules"));
            allPos.addAll((List<Object>)pos);
        }
        oneClass.put("allPos", ((List<Object>)allPos).stream().map(s -> s.toString()).collect(Collectors.toList()));
        oneClass.put("field", field);
        
        return oneClass;
        
    }

    private List<Object> writeRule(Object docId, int line_count, int i, Map<String, Object> rule) throws IOException {
        final Map<String, Object> oneRule = new HashMap<>();
        oneRule.put("score", rule.get("score"));
        oneRule.put("checks", new ArrayList<>());
        
        final List<Integer> allPos = new ArrayList<>();
        String field = "";
        
        for (Map<String, Object> check : ((List<Map<String,  Object>>)rule.get("checks"))) {
            final Map<String, Object> checkOneRule = new HashMap<>();
             List<Integer> pos = new ArrayList<>();
            
            if (!(new Double(0.0)).equals(check.get("feature value")) 
                    && ((Map<String, Object>)check.get("feature")).containsKey("ngram")) {
                pos = getPositions(docId, ((Map<String, Object>)check.get("feature")).get("field"), 
                        ((Map<String, Object>)check.get("feature")).get("ngram"), 
                        ((Map<String, Object>)check.get("feature")).get("slop"), 
                        ((Map<String, Object>)check.get("feature")).get("inOrder"));
                
                if (((Map<String, Object>)check.get("feature")).containsKey("ngram")) {
                    checkOneRule.put("name", ((Map<String, Object>)check.get("feature")).get("name"));
                    checkOneRule.put("index", ((Map<String, Object>)check.get("feature")).get("index"));
                } else {
                    checkOneRule.put("ngram", ((Map<String, Object>)check.get("feature")).get("ngram"));
                    checkOneRule.put("index", ((Map<String, Object>)check.get("feature")).get("index"));
                    checkOneRule.put("field", ((Map<String, Object>)check.get("feature")).get("field"));
                    checkOneRule.put("slop", ((Map<String, Object>)check.get("feature")).get("slop"));
                    field = (String) checkOneRule.get("field");
                }
                
                checkOneRule.put("value",    check.get("feature value"));
                checkOneRule.put("relation", check.get("relation"));
                checkOneRule.put("threshold",check.get("threshold"));
                checkOneRule.put("highlights", pos.stream().map(q -> String.valueOf(q)).collect(Collectors.toList()));
                
                allPos.addAll(pos);
                ((List<Object>)oneRule.get("checks")).add(checkOneRule);
                oneRule.put("checks", oneRule.get("checks"));
            }
            
        }
        
        return Arrays.asList(oneRule, allPos, field);
        
    }

    private List<Object> newSpanTerm(String terms, String field) {
        List<Object> allSpans = new ArrayList<>();
        
        for (String term : terms.split("\\s+")) {
            final Map<String, Object> spanTerm = new HashMap<>();
            final Map<String, Object> internalSpanTerm = new HashMap<>();
            internalSpanTerm.put(field, term);

            spanTerm.put("span_term", internalSpanTerm);
            allSpans.add(spanTerm);
        }
        
        return allSpans;
    }
    private List<Integer> getPositions(Object docId, Object field, Object keywords, Object slop, Object in_order) throws IOException {
        System.out.println(docId + " " + field + " " + keywords + " " + slop + " " + in_order);
        
        final URL url = new URL("http://localhost:9200/ohsumed_20000/document/_search");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.addRequestProperty("Content-Type", "application/" + "POST");
        
        final String clauses =getJsonString(newSpanTerm((String)keywords, (String)field));
        final String body = "{\"explain\":\"false\"," + 
                "                              \"query\":{" + 
                "                                  \"filtered\":{" + 
                "                                      \"query\":{" + 
                "                                          \"span_near\": {" + 
                "                                              \"clauses\":" + clauses + "," + 
                "                                              \"slop\":" + slop.toString() +  "," + 
                "                                              \"in_order\":"  + in_order.toString() + "," +  
                "                                              \"collect_payloads\": false}}," + 
                "                                      \"filter\":{\"ids\":{\"values\":[\"" + docId  + "\"]}}}}," + 
                "                              \"highlight\":{\"fields\":{\"" + field + "\":{}}}," + 
                "                              \"size\":1}";
        System.out.println(body);
        conn.setRequestProperty("Content-Length", Integer.toString(body.length()));
        conn.getOutputStream().write(body.getBytes("UTF-8"));
        
        final BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        final StringBuilder result = new StringBuilder();
        for (String line = null; (line = reader.readLine()) != null; ) {
           result.append(line);
        }
        
        final JsonFactory factory = new JsonFactory(); 
        final ObjectMapper mapper = new ObjectMapper(factory);
        
        final Map<String, Object> hits = mapper.readValue(result.toString(), new TypeReference<Map<String, Object>>() {
        });
        final List<Integer> positions = new ArrayList<>();
        
        if (hits.get("hits") != null && 
                ((Map<String, Object>) hits.get("hits")).containsKey("hits")) {
            for (String hlField : 
                ((Map<String, Object>)((Map<String, Object>)((List<Map<String, Object>>)((Map<String, Object>) hits.get("hits")).get("hits")).get(0)).get("highlight")).keySet()) {
                
                final String text = (String) ((Map<String, Object>)(((List<Map<String, Object>>)
                        ((Map<String, Object>) hits.get("hits")).get("hits")).get(0).get("_source"))).get(hlField);
                final List<String> highlights = (List<String>) ((Map<String, Object>)(((List<Map<String, Object>>)
                        ((Map<String, Object>) hits.get("hits")).get("hits")).get(0).get("highlight"))).get(hlField);
                
                for (String HL : highlights) {
                    String cleanHL = HL.replaceAll("<em>", "");
                    
                    cleanHL = cleanHL.replaceAll("</em>", "");
                    int baseindex = text.indexOf(cleanHL);

                    // in case the highlight not found in body
                    if (baseindex == -1) {
                                continue;
                    }

                     while (HL.indexOf("<em>") != -1) {
                         int start = HL.indexOf("<em>") + baseindex;
                         HL = HL.replace("<em>", "");
                         int end = HL.indexOf("</em>") + baseindex;
                         HL = HL.replace("</em>", "");
                         positions.add(start);
                         positions.add(end);
                     }
                }
                
                
            }
        }
        
        return positions; 
    }

    private Object includesLabel(Object label, List<Object> labels) {
        for (Object lbObj : labels) {
            Map<String, Object> lb = (Map<String, Object>)lbObj;
            for (Object key : lb.keySet()) {
                if (label.equals(lb.get(key))) {
                    return 1;
                }
            }
        }
        return 0;
    }

    private void createMetaDataHTML(File inputData, File inputModel, File inputConfig, File outputPath) throws IOException {
        Object inputD = null;
        Object inputM = null;
        Object inputC = null;
        
        final Map<String, Object> outputData = new HashMap<>();
        
        if (inputData.isFile()) {
            inputD = loadJson(inputData, new TypeReference<Object>() {});
        }
        
        if (inputModel.isFile()) {
            inputM = loadJson(inputData, new TypeReference<Object>() {});
        }
        
        if (inputConfig.isFile()) {
            inputC = loadJson(inputData, new TypeReference<Object>() {});
        }
        
        outputData.put("data", inputD);
        outputData.put("model", inputM);
        outputData.put("config", inputC);
        
        final String output = getResourcesFileAsString("pre-md-data.html")
                                 + getJsonString(outputData)
                                 + getResourcesFileAsString("post-data.txt");
        writeStringToFile(outputPath, output);
        
    }

    private void createIndPerHTML(File inputIndPer, File inputAllPer, File outputPath) throws IOException {
        if (inputIndPer.isFile()) {
            final String output = getResourcesFileAsString("pre-ind-data-part1.html")
                                + "["
                                + readFileAsString(inputAllPer)
                                + "]"
                                + getResourcesFileAsString("pre-ind-data-part2.txt")
                                + readFileAsString(inputIndPer)
                                + getResourcesFileAsString("post-data.txt");
            writeStringToFile(outputPath, output);
        }
    }

    private void createTopFeatureHTML(File inputFile, File outputFile) throws IOException {
        String output = "";
        
        try {
            final List<Map<String, Object>> inputData = loadJson(inputFile, new TypeReference<List<Map<String,Object>>>() {});
            final List<Object> outputData = createNewJsonForTopFeatures(inputData);
            final String outputJson = getJsonString(outputData);
            
            output = getResourcesFileAsString("pre-tf-data.html") + outputJson + getResourcesFileAsString("post-data.txt");
        } catch (IOException e) {
            e.printStackTrace();
            output = getResourcesFileAsString("pre-tf-data.html") + getResourcesFileAsString("post-data.txt");
        }
        
        FileUtils.writeStringToFile(outputFile, output, "UTF-8");
    }
    
    private void writeStringToFile(File file, String data) throws IOException {
        FileUtils.writeStringToFile(file, data, "UTF-8");
    }
    
    private String readFileAsString(File file) throws IOException {
        return FileUtils.readFileToString(file, "UTF-8");
    }
    
    private List<Object> createNewJsonForTopFeatures(List<Map<String, Object>> inputData) {
        List<Object> outputData = new ArrayList<>();
        outputData.add(new ArrayList<>()); //classes
        outputData.add(new ArrayList<>()); //details
        
        Map<String, Object> indexes = new HashMap<>();
        
        for (Map<String, Object> clas : inputData) {
            final List<Object> feature = new ArrayList<>();
            
            feature.add(clas.get("classIndex"));
            feature.add(clas.get("className"));
            
            List<Object> fds = new ArrayList<>();
            
            for (Map<String, Object> fd : ((List<Map<String,  Object>>)clas.get("featureDistributions"))) {
                final List<Object> distribution = new ArrayList<>();
                
                final String name;
                if (((Map<String, Object>) fd.get("feature")).containsKey("ngram")) {
                    name = (String) ((Map<String, Object>)fd.get("feature")).get("ngram");
                } else {
                    name = (String) ((Map<String, Object>)fd.get("feature")).get("name");
                }
                
                distribution.add(name);
                distribution.add(new ArrayList<>());
                distribution.add(fd.get("totalCount"));
                
                for (String occu : (List<String>)fd.get("occurrence")) {
                    final List<Object> occurence = new ArrayList<>();
                    
                    final String res[] = occu.split(":");
                    final String className = res[0];
                    final String r[] = res[1].split("/");
                    
                    if (!indexes.containsKey(className)) {
                        List<Object> c = new ArrayList<>();
                        c.add(className);
                        c.add(r[1]);
                        ((List<Object>) outputData.get(0)).add(c);
                        indexes.put(className, ((List<Object>) outputData.get(0)).size() - 1);
                        
                    }
                    occurence.add(indexes.get(className)); //classIndex
                    occurence.add(r[0]);
                    ((List<Object>) distribution.get(1)).add(occurence);
                }
                fds.add(distribution);
            }
            
            feature.add(fds);
            ((List<Object>) outputData.get(1)).add(feature);
        }
        
        return outputData;
    }

    private <T> T loadJson(File file, TypeReference<T> typeRef) throws IOException {
        final JsonFactory factory = new JsonFactory(); 
        final ObjectMapper mapper = new ObjectMapper(factory);
        
        return mapper.readValue(file, typeRef);
    }
    
    private void storeJson(File jsonFile, Map<?, ?> data) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        
        mapper.writeValue(jsonFile, data);
    }
    
    private String getJsonString(Object data) throws IOException {
        return new ObjectMapper().writeValueAsString(data);
    }
    
    private String getResourcesFileAsString(String fileName) throws IOException {
        return FileUtils.readFileToString(new File(config.getResourcesDir(), fileName), "UTF-8");
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.err.println("Requires name of the config file as the first argument");
            System.exit(1);
        }
        
        Visualizer visualizer = new Visualizer(new VisualizerConfig(new Config(args[0])));
        visualizer.visualize();
    }

}
