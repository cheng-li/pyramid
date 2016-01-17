package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.feature.Ngram;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Created by Rainicy on 8/3/15.
 */
public class MekaFormat {

    public static MultiLabelClfDataSet loadMLClfDataset(String fileName, int numFeatures, int numClasses, String dataMode) throws IOException {

        if (dataMode.equals("sparse")) {
            return loadMLClfDataset(new File(fileName), numFeatures, numClasses);
        } else if (dataMode.equals("dense")) {
            return loadMLClfDatasetDense(new File(fileName), numFeatures, numClasses);
        } else if (dataMode.equals("pre.dense")) {
            return loadMLClfDatasetPreDense(new File(fileName), numFeatures, numClasses);
        } else {
            throw new RuntimeException("not acceptable data mode: " + dataMode);
        }
    }

    private static MultiLabelClfDataSet loadMLClfDatasetPreDense(File file, int numFeatures, int numClasses) throws IOException {
        int numData = 0;
        Map<String, String> labelMap = new HashMap<>();
        Map<String, String> featureMap = new HashMap<>();
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        int featureCount = -1;
        boolean ifData = false;
        while((line=br.readLine())!=null) {
            if (line.startsWith("@data")) {
                ifData = true;
                continue;
            }
            if (featureCount < numFeatures) {
                if (line.startsWith("@attribute")) {
                    if (featureCount==-1) {
                        featureCount++;
                        continue;
                    }
                    String[] splitLine = line.split(" ");
                    String featureName = splitLine[1];
                    String featureIndex = Integer.toString(featureCount);
                    featureMap.put(featureIndex, featureName);
                    featureCount++;
                }
            }
            else {
                if (line.startsWith("@attribute")) {
                    String[] splitLine = line.split(" ");
                    String labelName = splitLine[1];
                    String labelIndex = Integer.toString(featureCount);
                    labelMap.put(labelIndex, labelName);
                    featureCount++;
                } else if (ifData && line.length() >= 2) {
                    numData++;
                }
            }
        }
        br.close();

        return loadMLClfDatasetPreDense(file, numClasses, numFeatures, numData, labelMap, featureMap);
    }

    private static MultiLabelClfDataSet loadMLClfDatasetPreDense(File file, int numClasses, int numFeatures, int numData,
                                                                 Map<String, String> labelMap, Map<String, String> featureMap) throws IOException {
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numData).numClasses(numClasses).numFeatures(numFeatures)
                .build();

        // set features
        List<Feature> featureList = new LinkedList<>();
        for (int m=0; m<numFeatures; m++) {
            String featureIndex = Integer.toString(m);
            String featureName = featureMap.get(featureIndex);
            Feature feature = new Feature();
            feature.setIndex(m);
            feature.setName(featureName);
            featureList.add(feature);
        }
        dataSet.setFeatureList(new FeatureList(featureList));
        // set Label
        Map<Integer, String> labelIndexMap = new HashMap<>();
        for (Map.Entry<String, String> entry : labelMap.entrySet()) {
            String labelString = entry.getKey();
            String labelName = entry.getValue();
            labelIndexMap.put(Integer.parseInt(labelString)-numFeatures, labelName);
        }
        LabelTranslator labelTranslator = new LabelTranslator(labelIndexMap);
        dataSet.setLabelTranslator(labelTranslator);

        // create feature matrix
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        int dataCount = 0;
        boolean ifData = false;
        while((line=br.readLine()) != null) {
            if (line.startsWith("@data")) {
                ifData = true;
                continue;
            }
            if (ifData) {
                if ((line.startsWith("{")) && (line.endsWith("}"))) {
                    line = line.substring(1,line.length()-1);
                }
                String[] indexValues = line.split(",");
                int indexValueI = -2;
                for (String indexValue : indexValues) {
                    indexValueI++;
                    if (indexValueI == -1) {
                        continue;
                    }
                    String index = Integer.toString(indexValueI);
                    String value = indexValue;
                    if (labelMap.containsKey(index)) {
                        double valueDouble = Double.parseDouble(value);
                        if (valueDouble == 1.0) {
                            dataSet.addLabel(dataCount, Integer.parseInt(index)-numFeatures);
                        }
                    } else if (featureMap.containsKey(index)) {
                        double valueDouble = Double.parseDouble(value);
                        int indexInt = Integer.parseInt(index);
                        dataSet.setFeatureValue(dataCount,indexInt,valueDouble);
                    } else {
                        throw new RuntimeException("Index not found in the line: " + line);
                    }
                }
                dataCount++;
            }
        }
        br.close();
        return dataSet;
    }

    private static MultiLabelClfDataSet loadMLClfDatasetDense(File file, int numFeatures, int numClasses) throws IOException {

        int numData = 0;
        Map<String, String> labelMap = new HashMap<>();
        Map<String, String> featureMap = new HashMap<>();
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        int featureCount = 0;
        boolean ifData = false;
        while((line=br.readLine())!=null) {
            if (line.startsWith("@data")) {
                ifData = true;
                continue;
            }
            if (featureCount < numFeatures) {
                if (line.startsWith("@attribute")) {
                    String[] splitLine = line.split(" ");
                    String featureName = splitLine[1];
                    String featureIndex = Integer.toString(featureCount);
                    featureMap.put(featureIndex, featureName);
                    featureCount++;
                }
            }
            else {
                if (line.startsWith("@attribute")) {
                    String[] splitLine = line.split(" ");
                    String labelName = splitLine[1];
                    String labelIndex = Integer.toString(featureCount);
                    labelMap.put(labelIndex, labelName);
                    featureCount++;
                } else if (ifData && line.length() >= 2) {
                    numData++;
                }
            }
        }
        br.close();

        return loadMLClfDatasetDense(file, numClasses, numFeatures, numData, labelMap, featureMap);
    }

    private static MultiLabelClfDataSet loadMLClfDatasetDense(File file, int numClasses, int numFeatures, int numData, Map<String, String> labelMap, Map<String, String> featureMap) throws IOException {
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numData).numClasses(numClasses).numFeatures(numFeatures)
                .build();

        // set features
        List<Feature> featureList = new LinkedList<>();
        for (int m=0; m<numFeatures; m++) {
            String featureIndex = Integer.toString(m);
            String featureName = featureMap.get(featureIndex);
            Feature feature = new Feature();
            feature.setIndex(m);
            feature.setName(featureName);
            featureList.add(feature);
        }
        dataSet.setFeatureList(new FeatureList(featureList));
        // set Label
        Map<Integer, String> labelIndexMap = new HashMap<>();
        for (Map.Entry<String, String> entry : labelMap.entrySet()) {
            String labelString = entry.getKey();
            String labelName = entry.getValue();
            labelIndexMap.put(Integer.parseInt(labelString)-numFeatures, labelName);
        }
        LabelTranslator labelTranslator = new LabelTranslator(labelIndexMap);
        dataSet.setLabelTranslator(labelTranslator);

        // create feature matrix
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        int dataCount = 0;
        boolean ifData = false;
        while((line=br.readLine()) != null) {
            if (line.startsWith("@data")) {
                ifData = true;
                continue;
            }
            if (ifData) {
                if ((line.startsWith("{")) && (line.endsWith("}"))) {
                    line = line.substring(1,line.length()-1);
                }
                String[] indexValues = line.split(",");
                int indexValueI = -1;
                for (String indexValue : indexValues) {
                    indexValueI++;
                    String index = Integer.toString(indexValueI);
                    String value = indexValue;
                    if (labelMap.containsKey(index)) {
                        double valueDouble = Double.parseDouble(value);
                        if (valueDouble == 1.0) {
                            dataSet.addLabel(dataCount, Integer.parseInt(index)-numFeatures);
                        }
                    } else if (featureMap.containsKey(index)) {
                        double valueDouble = Double.parseDouble(value);
                        int indexInt = Integer.parseInt(index);
                        dataSet.setFeatureValue(dataCount,indexInt,valueDouble);
                    } else {
                        throw new RuntimeException("Index not found in the line: " + line);
                    }
                }
                dataCount++;
            }
        }
        br.close();
        return dataSet;
    }

    private static MultiLabelClfDataSet loadMLClfDataset(File file, int numFeatures, int numClasses) throws IOException {

        int numData = 0;
        Map<String, String> labelMap = new HashMap<>();
        Map<String, String> featureMap = new HashMap<>();
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        int featureCount = 0;
        while((line=br.readLine())!=null) {
            if (featureCount < numFeatures) {
                if (line.startsWith("@attribute")) {
                    String[] splitLine = line.split(" ");
                    String featureName = splitLine[1];
                    String featureIndex = Integer.toString(featureCount);
                    featureMap.put(featureIndex, featureName);
                    featureCount++;
                }
            }
            else {
                if (line.startsWith("@attribute")) {
                    String[] splitLine = line.split(" ");
                    String labelName = splitLine[1];
                    String labelIndex = Integer.toString(featureCount);
                    labelMap.put(labelIndex, labelName);
                    featureCount++;
                } else if ((line.startsWith("{")) && (line.endsWith("}"))) {
                    numData++;
                }
            }
        }
        br.close();

        return loadMLClfDataset(file, numClasses, numFeatures, numData, labelMap, featureMap);

    }

    private static MultiLabelClfDataSet loadMLClfDataset(File file, int numClasses, int numFeatures, int numData,
                                                         Map<String, String> labelMap, Map<String, String> featureMap) throws IOException {
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numData).numClasses(numClasses).numFeatures(numFeatures)
                .build();

        // set features
        List<Feature> featureList = new LinkedList<>();
        for (int m=0; m<numFeatures; m++) {
            String featureIndex = Integer.toString(m);
            String featureName = featureMap.get(featureIndex);
            Feature feature = new Feature();
            feature.setIndex(m);
            feature.setName(featureName);
            featureList.add(feature);
        }
        dataSet.setFeatureList(new FeatureList(featureList));
        // set Label
        Map<Integer, String> labelIndexMap = new HashMap<>();
        for (Map.Entry<String, String> entry : labelMap.entrySet()) {
            String labelString = entry.getKey();
            String labelName = entry.getValue();
            labelIndexMap.put(Integer.parseInt(labelString)-numFeatures, labelName);
        }
        LabelTranslator labelTranslator = new LabelTranslator(labelIndexMap);
        dataSet.setLabelTranslator(labelTranslator);

        // create feature matrix
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        int dataCount = 0;
        while((line=br.readLine()) != null) {
            if ((line.startsWith("{")) && (line.endsWith("}"))) {
                line = line.substring(1,line.length()-1);
                String[] indexValues = line.split(",");
                for (String indexValue : indexValues) {
                    String[] indexValuePair = indexValue.split(" ");
                    String index = indexValuePair[0];
                    String value = indexValuePair[1];
                    if (labelMap.containsKey(index)) {
                        double valueDouble = Double.parseDouble(value);
                        if (valueDouble == 1.0) {
                            dataSet.addLabel(dataCount, Integer.parseInt(index)-numFeatures);
                        }
                    } else if (featureMap.containsKey(index)) {
                        double valueDouble = Double.parseDouble(value);
                        int indexInt = Integer.parseInt(index);
                        dataSet.setFeatureValue(dataCount,indexInt,valueDouble);
                    } else {
                        throw new RuntimeException("Index not found in the line: " + line);
                    }
                }
                dataCount++;
            }
        }
        br.close();
        return dataSet;
    }

    public static void save(MultiLabelClfDataSet dataSet, String mekaFile, Config config) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(mekaFile));

        // writing the header: @relation 'data_name: -C number_classes\n\n'
        String dataName = config.getString("data.name");
        // starting writing labels
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        int numClasses = labelTranslator.getNumClasses();
        bw.write("@relation " + "'" + dataName + ": -C " + numClasses + "'\n\n");
        for (int i=0; i<numClasses; i++) {
            String labelName = labelTranslator.toExtLabel(i);
            bw.write("@attribute " + labelName.replace(" ", "_") + " {0,1}\n");
        }

        // starting writing features
        FeatureList featureList = dataSet.getFeatureList();
        Pattern pattern = Pattern.compile("ngram=(.*?), field");
        for (int i=0; i<featureList.size(); i++) {
            Feature feature = featureList.get(i);
            String featureName = "";
            if (feature instanceof Ngram) {
                Ngram ngram = (Ngram) feature;
                featureName = ngram.getNgram();
            }
            bw.write("@attribute " + featureName + " numeric\n");
        }

        // starting @data
        MultiLabel[] multiLabels = dataSet.getMultiLabels();
        bw.write("\n@data\n\n");
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            StringBuffer stringBuffer = new StringBuffer();
            stringBuffer.append("{");
            Vector rowData = dataSet.getRow(i);
            MultiLabel multiLabel = multiLabels[i];

            //starting with labels index.
            List<Integer> matchedLabels = multiLabel.getMatchedLabelsOrdered();
            for (int j=0; j<matchedLabels.size(); j++) {
                int matchedLabel = matchedLabels.get(j);
                if (j != 0) {
                    stringBuffer.append(",");
                }
                stringBuffer.append(matchedLabel + " " + "1");
            }
            // following by feature index
            Map<Integer, Double> sortedKeys = new TreeMap<>();
            for (Vector.Element element : rowData.nonZeroes()) {
                int index = element.index();
                double value = element.get();
                sortedKeys.put(index, value);
            }
            for (Map.Entry<Integer, Double> entry : sortedKeys.entrySet()) {
                int index = entry.getKey();
                double value = entry.getValue();
//                System.out.println("old index: " + index);
                index += numClasses;
//                System.out.println("new index: " + index);
//                System.in.read();
                stringBuffer.append("," + index + " " + value);
            }
            stringBuffer.append("}\n");
            bw.write(stringBuffer.toString());
        }
        bw.close();
    }
}
