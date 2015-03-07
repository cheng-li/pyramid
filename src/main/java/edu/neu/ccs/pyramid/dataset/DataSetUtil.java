package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.mahout.math.Vector;


import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/7/14.
 */
public class DataSetUtil {

    /**
     *
     * @param dataSet
     * @param numClasses for new dataset
     * @return
     */
    public static ClfDataSet changeLabels(ClfDataSet dataSet, int numClasses){
        ClfDataSet dataSet1;
        int numDataPoints = dataSet.getNumDataPoints();
        int numFeatures = dataSet.getNumFeatures();
        boolean missingValue = dataSet.hasMissingValue();
        if (dataSet.isDense()){
            dataSet1 = new DenseClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        } else {
            dataSet1 = new SparseClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        }
        for (int i=0;i<numDataPoints;i++){
            //only copy non-zero elements
            Vector vector = dataSet.getRow(i);
            for (Vector.Element element: vector.nonZeroes()){
                int featureIndex = element.index();
                double value = element.get();
                if (featureIndex<numFeatures){
                    dataSet1.setFeatureValue(i,featureIndex,value);
                }
            }
        }

        return dataSet1;
    }

    /**
     * only keep the selected featureList
     * @param dataSet
     * @return
     */
    public static ClfDataSet trim(ClfDataSet dataSet, List<Integer> columnsToKeep){
        ClfDataSet trimmed ;
        int numClasses = dataSet.getNumClasses();
        boolean missingValue = dataSet.hasMissingValue();
        // keep density
        if (dataSet.isDense()) {
            trimmed = new DenseClfDataSet(dataSet.getNumDataPoints(), columnsToKeep.size(), missingValue, numClasses);
        } else{
            trimmed = new SparseClfDataSet(dataSet.getNumDataPoints(),columnsToKeep.size(), missingValue, numClasses);
        }


        for (int j=0;j<trimmed.getNumFeatures();j++){
            int oldColumnIndex = columnsToKeep.get(j);
            Vector vector = dataSet.getColumn(oldColumnIndex);
            for (Vector.Element element: vector.nonZeroes()){
                int dataPointIndex = element.index();
                double value = element.get();
                trimmed.setFeatureValue(dataPointIndex,j,value);
            }
        }
        //copy labels
        int[] labels = dataSet.getLabels();
        for (int i=0;i<trimmed.getNumDataPoints();i++){
            trimmed.setLabel(i,labels[i]);
        }

        trimmed.setLabelTranslator(dataSet.getLabelTranslator());
        trimmed.setIdTranslator(dataSet.getIdTranslator());
        List<Feature> oldFeatures = dataSet.getFeatureList().getAll();
        List<Feature> newFeatures = columnsToKeep.stream().map(oldFeatures::get).collect(Collectors.toList());
        for (int i=0;i<newFeatures.size();i++){
            newFeatures.get(i).setIndex(i);
        }
        trimmed.setFeatureList(new FeatureList(newFeatures));
        return trimmed;
    }


    /**
     * only keep the selected featureList
     * @param dataSet
     * @return
     */
    public static MultiLabelClfDataSet trim(MultiLabelClfDataSet dataSet, List<Integer> columnsToKeep){
        MultiLabelClfDataSet trimmed ;
        boolean missingValue = dataSet.hasMissingValue();
        int numClasses = dataSet.getNumClasses();
        // keep density
        if (dataSet.isDense()) {
            trimmed = new DenseMLClfDataSet(dataSet.getNumDataPoints(), columnsToKeep.size(), missingValue, numClasses);
        } else{
            trimmed = new SparseMLClfDataSet(dataSet.getNumDataPoints(),columnsToKeep.size(), missingValue, numClasses);
        }


        for (int j=0;j<trimmed.getNumFeatures();j++){
            int oldColumnIndex = columnsToKeep.get(j);
            Vector vector = dataSet.getColumn(oldColumnIndex);
            for (Vector.Element element: vector.nonZeroes()){
                int dataPointIndex = element.index();
                double value = element.get();
                trimmed.setFeatureValue(dataPointIndex,j,value);
            }
        }
        //copy labels
        MultiLabel[] multiLabels = dataSet.getMultiLabels();

        for (int i=0;i<trimmed.getNumDataPoints();i++){
            trimmed.addLabels(i,multiLabels[i].getMatchedLabels());
        }
        //just copy settings


        trimmed.setLabelTranslator(dataSet.getLabelTranslator());
        trimmed.setIdTranslator(dataSet.getIdTranslator());
        List<Feature> oldFeatures = dataSet.getFeatureList().getAll();
        List<Feature> newFeatures = columnsToKeep.stream().map(oldFeatures::get).collect(Collectors.toList());
        for (int i=0;i<newFeatures.size();i++){
            newFeatures.get(i).setIndex(i);
        }
        trimmed.setFeatureList(new FeatureList(newFeatures));

        return trimmed;
    }

    /**
     * only keep the first numFeatures featureList
     * @param clfDataSet
     * @param numFeatures
     * @return
     */
    public static ClfDataSet trim(ClfDataSet clfDataSet, int numFeatures){
        List<Integer> columnsToKeep = IntStream.range(0,numFeatures).mapToObj(i->i).collect(Collectors.toList());
        return  trim(clfDataSet,columnsToKeep);
    }

    /**
     *
     * @param inputFile
     * @param outputFile
     * @param start inclusive, first column is 0
     * @param end inclusive
     */
    public static void extractColumns(String inputFile, String outputFile,
                                      int start, int end, String delimiter) throws IOException{
        try(BufferedReader br = new BufferedReader(new FileReader(new File(inputFile)));
            BufferedWriter bw = new BufferedWriter(new FileWriter(new File(outputFile)))
        ){
            String line;
            while((line = br.readLine())!=null){
                String[] split = line.split(Pattern.quote(delimiter));
                System.out.println(Arrays.toString(split));
                System.out.println(split.length);
                for (int i=start;i<=end;i++){
                    System.out.println(i);
                    bw.write(split[i]);
                    if (i<end){
                        bw.write(delimiter);
                    }
                }
                bw.newLine();
            }
        }
    }

    public static void extractColumns(String inputFile, String outputFile,
                                      int start, int end, Pattern pattern) throws IOException{
        try(BufferedReader br = new BufferedReader(new FileReader(new File(inputFile)));
            BufferedWriter bw = new BufferedWriter(new FileWriter(new File(outputFile)))
        ){
            String line;
            while((line = br.readLine())!=null){
                //remove leading spaces
                String[] split = line.trim().split(pattern.pattern());
                for (int i=start;i<=end;i++){
                    bw.write(split[i]);
                    if (i<end){
                        bw.write(",");
                    }
                }
                bw.newLine();
            }
        }
    }


//    /**
//     *
//     * @param dataSet
//     * @param extLabels in order
//     */
//    public static void setLabelTranslator(ClfDataSet dataSet, List<String> extLabels){
//        int[] labels = dataSet.getLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            dataSet.getRow(i).getSettings()
//                    .setExtLabel(extLabels.get(labels[i]));
//        }
//        dataSet.getSettings().setLabelTranslator(new LabelTranslator(extLabels));
//    }
//
//    /**
//     *
//     * @param dataSet
//     * @param extLabels in order
//     */
//    public static void setLabelTranslator(ClfDataSet dataSet, String[] extLabels){
//        int[] labels = dataSet.getLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            dataSet.getRow(i).getSettings()
//                    .setExtLabel(extLabels[labels[i]]);
//        }
//        dataSet.getSettings().setLabelTranslator(new LabelTranslator(extLabels));
//    }
//
//    public static void setLabelTranslator(ClfDataSet dataSet, Map<Integer, String> intToExtLabel){
//        int[] labels = dataSet.getLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            dataSet.getRow(i).getSettings()
//                    .setExtLabel(intToExtLabel.get(labels[i]));
//        }
//        dataSet.getSettings().setLabelTranslator(new LabelTranslator(intToExtLabel));
//    }


//    public static void setLabelTranslator(MultiLabelClfDataSet dataSet, Map<Integer, String> intToExtLabel){
//        MultiLabel[] multiLabels= dataSet.getMultiLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            MultiLabel multiLabel = multiLabels[i];
//            List<String> extLabels = multiLabel.getMatchedLabels().stream()
//                    .map(intToExtLabel::get)
//                    .collect(Collectors.toList());
//            dataSet.getRow(i).getSettings()
//                    .setExtLabels(extLabels);
//        }
//        dataSet.getSettings().setLabelTranslator(new LabelTranslator(intToExtLabel));
//    }
//
//    public static void setLabelTranslator(MultiLabelClfDataSet dataSet, String[] extLabels){
//        MultiLabel[] multiLabels= dataSet.getMultiLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            MultiLabel multiLabel = multiLabels[i];
//            List<String> matchedExtLabels = multiLabel.getMatchedLabels().stream()
//                    .map(label -> extLabels[label])
//                    .collect(Collectors.toList());
//            dataSet.getRow(i).getSettings()
//                    .setExtLabels(matchedExtLabels);
//        }
//        dataSet.getSettings().setLabelTranslator(new LabelTranslator(extLabels));
//    }



    /**
     * stratified bootstrap sample
     * each class has the same number of data points as in the original data set
     * @param clfDataSet
     * @return
     */
    public static ClfDataSet bootstrap(ClfDataSet clfDataSet){
        Map<Integer, List<Integer>> labelIndicesMap = new HashMap<>();
        int[] labels = clfDataSet.getLabels();
        for (int i=0;i<clfDataSet.getNumDataPoints();i++){
            int label = labels[i];
            if (!labelIndicesMap.containsKey(label)){
                labelIndicesMap.put(label, new ArrayList<>());
            }
            labelIndicesMap.get(label).add(i);
        }
        List<Integer> sampledIndices = new ArrayList<>(clfDataSet.getNumDataPoints());
        //sample for each class
        for (Map.Entry<Integer,List<Integer>> entry: labelIndicesMap.entrySet()) {
            List<Integer> indices = entry.getValue();
            int[] sampleForClass = Sampling.sampleWithReplacement(indices.size(), indices).toArray();
            for (int index: sampleForClass){
                sampledIndices.add(index);
            }
        }

        return subSet(clfDataSet,sampledIndices);
    }

    /**
     * create a subset with the indices
     * it's fine to have duplicate indices
     * idTranslator is not saved in subSet as we may have duplicate extIds
     * @param dataSet
     * @param indices
     * @return
     */
    public static ClfDataSet subSet(ClfDataSet dataSet, List<Integer> indices){
        ClfDataSet sample;
        int numClasses = dataSet.getNumClasses();
        boolean missingValue = dataSet.hasMissingValue();
        if (dataSet instanceof DenseClfDataSet){
            sample = new DenseClfDataSet(indices.size(),dataSet.getNumFeatures(), missingValue, numClasses);
        } else {
            sample = new SparseClfDataSet(indices.size(),dataSet.getNumFeatures(), missingValue, numClasses);
        }
        int[] labels = dataSet.getLabels();
        for (int i=0;i<indices.size();i++){
            int indexInOld = indices.get(i);
            Vector oldVector = dataSet.getRow(indexInOld);
            int label = labels[indexInOld];
            //copy label
            sample.setLabel(i,label);
            //copy row feature values, optimized for sparse vector
            for (Vector.Element element: oldVector.nonZeroes()){
                sample.setFeatureValue(i,element.index(),element.get());
            }

        }

        sample.setLabelTranslator(dataSet.getLabelTranslator());
        sample.setFeatureList(dataSet.getFeatureList());

        //ignore idTranslator as we may have duplicate extIds
        return sample;
    }

    /**
     *
     * @param clfDataSet
     * @return training set and validation set
     */
    public static Pair<ClfDataSet,ClfDataSet> splitToTrainValidation(ClfDataSet clfDataSet, double trainPercentage){
        int numDataPoints = clfDataSet.getNumDataPoints();
        List<Integer> trainIndices = Sampling.stratified(clfDataSet.getLabels(),trainPercentage);

        Set<Integer> testIndicesSet = new HashSet<>();
        for (int i=0;i<numDataPoints;i++){
            testIndicesSet.add(i);
        }
        testIndicesSet.removeAll(trainIndices);
        List<Integer> testIndices = testIndicesSet.stream().collect(Collectors.toList());
        Pair<ClfDataSet,ClfDataSet> pair = new Pair<>();
        pair.setFirst(DataSetUtil.subSet(clfDataSet,trainIndices));
        pair.setSecond(DataSetUtil.subSet(clfDataSet,testIndices));
        return pair;
    }

    public static void dumpDataPointSettings(ClfDataSet dataSet, String file) throws IOException{
        dumpDataPointSettings(dataSet, new File(file));
    }

    /**
     * dump data settings to a plain text file
     * @param dataSet
     * @param file
     * @throws IOException
     */
    public static void dumpDataPointSettings(ClfDataSet dataSet, File file) throws IOException {
        int numDataPoints = dataSet.getNumDataPoints();
        int[] labels = dataSet.getLabels();
        IdTranslator idTranslator = dataSet.getIdTranslator();
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))
        ) {
            for (int i = 0; i < numDataPoints; i++) {
                bw.write("intId=");
                bw.write("" + i);
                bw.write(",");
                bw.write("extId=");
                bw.write(idTranslator.toExtId(i));
                bw.write(",");
                bw.write("intLabel=");
                bw.write("" + labels[i]);
                bw.write(",");
                bw.write("extLabel=");
                bw.write(labelTranslator.toExtLabel(labels[i]));
                bw.newLine();
            }
        }
    }

    public static void dumpDataPointSettings(MultiLabelClfDataSet dataSet, String file) throws IOException{
        dumpDataPointSettings(dataSet, new File(file));
    }

    public static void dumpDataPointSettings(MultiLabelClfDataSet dataSet, File file) throws IOException {
        IdTranslator idTranslator = dataSet.getIdTranslator();
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        int numDataPoints = dataSet.getNumDataPoints();
        MultiLabel[] labels = dataSet.getMultiLabels();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))
        ) {
            for (int i = 0; i < numDataPoints; i++) {

                bw.write("intId=");
                bw.write("" + i);
                bw.write(",");
                bw.write("extId=");
                bw.write(idTranslator.toExtId(i));
                bw.write(",");
                bw.write("intLabel=");
                bw.write("" + labels[i].getMatchedLabelsOrdered());
                bw.write(",");
                bw.write("extLabel=");
                bw.write(labels[i].getMatchedLabelsOrdered().stream()
                        .map(labelTranslator::toExtLabel).collect(Collectors.toList()).toString());
                bw.newLine();
            }
        }
    }

    /**
     * dump feature settings to plain text file
     * @param dataSet
     * @param file
     * @throws IOException
     */
    public static void dumpFeatureSettings(DataSet dataSet, String file) throws IOException {
        dumpFeatureSettings(dataSet,new File(file));
    }
    public static void dumpFeatureSettings(DataSet dataSet, File file) throws IOException {
        int numFeatures = dataSet.getNumFeatures();
        List<Feature> features = dataSet.getFeatureList().getAll();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))
        ) {
            for (int j=0;j<numFeatures;j++){
                bw.write(features.get(j).toString());
                bw.newLine();
            }
        }
    }


    public static List<MultiLabel> gatherLabels(MultiLabelClfDataSet dataSet){
        Set<MultiLabel> multiLabels = new HashSet<>();
        MultiLabel[] multiLabelsArray = dataSet.getMultiLabels();
        for (MultiLabel multiLabel: multiLabelsArray){
            multiLabels.add(multiLabel);
        }
        return multiLabels.stream().collect(Collectors.toList());
    }

    /**
     * merge to binary dataset
     * k=positive (1), others = negative(0)
     * @param dataSet
     * @param k
     * @return
     */
    public static ClfDataSet toBinary(MultiLabelClfDataSet dataSet, int k){
        int numDataPoints = dataSet.getNumDataPoints();
        int numFeatures = dataSet.getNumFeatures();
        boolean missingValue = dataSet.hasMissingValue();
        ClfDataSet clfDataSet;
        if (dataSet.isDense()){
            clfDataSet = new DenseClfDataSet(numDataPoints,numFeatures,missingValue, 2);
        } else {
            clfDataSet = new SparseClfDataSet(numDataPoints,numFeatures,missingValue, 2);
        }

        for (int i=0;i<numDataPoints;i++){
            //only copy non-zero elements
            Vector vector = dataSet.getRow(i);
            for (Vector.Element element: vector.nonZeroes()){
                int featureIndex = element.index();
                double value = element.get();
                clfDataSet.setFeatureValue(i,featureIndex,value);
            }
            if (dataSet.getMultiLabels()[i].matchClass(k)){
                clfDataSet.setLabel(i,1);
            } else {
                clfDataSet.setLabel(i,0);
            }
        }

        List<String> extLabels = new ArrayList<>();
        String extLabel = dataSet.getLabelTranslator().toExtLabel(k);
        extLabels.add("NOT "+extLabel);
        extLabels.add(extLabel);
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);
        clfDataSet.setLabelTranslator(labelTranslator);
        clfDataSet.setFeatureList(dataSet.getFeatureList());


        return clfDataSet;
    }


    public static void allowMissingValue(DataSet dataSet){
        if (dataSet instanceof AbstractDataSet){
            ((AbstractDataSet)dataSet).allowMissingValue();
        }
    }

    public static int[] getCountPerClass(ClfDataSet dataSet){
        int numClasses = dataSet.getNumClasses();
        int[] counts = new int[numClasses];
        int[] labels = dataSet.getLabels();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            int label = labels[i];
            counts[label] += 1;
        }
        return counts;
    }

    public static int[] getCountPerClass(MultiLabelClfDataSet dataSet){
        int numClasses = dataSet.getNumClasses();
        int[] counts = new int[numClasses];
        MultiLabel[] labels = dataSet.getMultiLabels();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            MultiLabel multiLabel = labels[i];
            for (int label: multiLabel.getMatchedLabels()){
                counts[label] += 1;
            }
        }
        return counts;
    }

    public static List<List<Integer>> labelToDataPoints(ClfDataSet dataSet){
        int numClasses = dataSet.getNumClasses();
        int[] labels = dataSet.getLabels();
        List<List<Integer>> list = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            list.add(new ArrayList<>());
        }
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            int label = labels[i];
            list.get(label).add(i);
        }
        return list;
    }


    public static List<List<Integer>> labelToDataPoints(MultiLabelClfDataSet dataSet){
        int numClasses = dataSet.getNumClasses();
        MultiLabel[] labels = dataSet.getMultiLabels();

        List<List<Integer>> list = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            list.add(new ArrayList<>());
        }
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            MultiLabel multiLabel = labels[i];
            for (int label: multiLabel.getMatchedLabels()){
                list.get(label).add(i);
            }
        }
        return list;
    }

    public static double density(DataSet dataSet){
        int nonZeros = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .map(i -> dataSet.getRow(i).getNumNonZeroElements())
                .sum();
        return ((double)nonZeros)/(dataSet.getNumDataPoints()*dataSet.getNumFeatures());
    }


}
