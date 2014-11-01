package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.feature.FeatureMappers;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.mahout.math.Vector;


import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

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
            FeatureRow featureRow = dataSet.getFeatureRow(i);
            //only copy non-zero elements
            Vector vector = featureRow.getVector();
            for (Vector.Element element: vector.nonZeroes()){
                int featureIndex = element.index();
                double value = element.get();
                if (featureIndex<numFeatures){
                    dataSet1.setFeatureValue(i,featureIndex,value);
                }
            }
        }
        for (int i=0;i<numDataPoints;i++){
            DataSetting dataSetting = dataSet.getFeatureRow(i).getSetting().copy();
            dataSetting.setExtLabel("unknown");
            dataSet1.getFeatureRow(i).putSetting(dataSetting);
        }

        for (int j=0;j<numFeatures;j++){
            FeatureSetting featureSetting = dataSet.getFeatureColumn(j).getSetting().copy();
            dataSet1.getFeatureColumn(j).putSetting(featureSetting);
        }


        return dataSet1;
    }

    /**
     * only keep the selected features
     * @param clfDataSet
     * @return
     */
    public static ClfDataSet trim(ClfDataSet clfDataSet, List<Integer> columnsToKeep){
        ClfDataSet trimmed ;
        int numClasses = clfDataSet.getNumClasses();
        boolean missingValue = clfDataSet.hasMissingValue();
        // keep density
        if (clfDataSet.isDense()) {
            trimmed = new DenseClfDataSet(clfDataSet.getNumDataPoints(), columnsToKeep.size(), missingValue, numClasses);
        } else{
            trimmed = new SparseClfDataSet(clfDataSet.getNumDataPoints(),columnsToKeep.size(), missingValue, numClasses);
        }


        for (int j=0;j<trimmed.getNumFeatures();j++){
            int oldColumnIndex = columnsToKeep.get(j);
            FeatureColumn featureColumn = clfDataSet.getFeatureColumn(oldColumnIndex);
            Vector vector = featureColumn.getVector();
            for (Vector.Element element: vector.nonZeroes()){
                int dataPointIndex = element.index();
                double value = element.get();
                trimmed.setFeatureValue(dataPointIndex,j,value);
            }
        }
        //copy labels
        int[] labels = clfDataSet.getLabels();
        for (int i=0;i<trimmed.getNumDataPoints();i++){
            trimmed.setLabel(i,labels[i]);
        }
        //just copy settings
        for (int i=0;i<trimmed.getNumDataPoints();i++){
            trimmed.getFeatureRow(i).putSetting(clfDataSet.getFeatureRow(i).getSetting().copy());
        }
        for (int j=0;j<trimmed.getNumFeatures();j++){
            int oldColumnIndex = columnsToKeep.get(j);
            trimmed.getFeatureColumn(j).putSetting(clfDataSet.getFeatureColumn(oldColumnIndex).getSetting().copy());
        }
        //todo double-check
        //todo: something like featuremappers
        DataSetSetting dataSetSetting = clfDataSet.getSetting().copy();
        dataSetSetting.setFeatureMappers(null);
        trimmed.putSetting(dataSetSetting);
        return trimmed;
    }


    /**
     * only keep the selected features
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
            FeatureColumn featureColumn = dataSet.getFeatureColumn(oldColumnIndex);
            Vector vector = featureColumn.getVector();
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
        for (int i=0;i<trimmed.getNumDataPoints();i++){
            trimmed.getFeatureRow(i).putSetting(dataSet.getFeatureRow(i).getSetting().copy());
        }
        for (int j=0;j<trimmed.getNumFeatures();j++){
            int oldColumnIndex = columnsToKeep.get(j);
            trimmed.getFeatureColumn(j).putSetting(dataSet.getFeatureColumn(oldColumnIndex).getSetting().copy());
        }
        //todo double-check
        //todo: something like featuremappers
        DataSetSetting dataSetSetting = dataSet.getSetting().copy();
        dataSetSetting.setFeatureMappers(null);
        trimmed.putSetting(dataSetSetting);
        return trimmed;
    }

    /**
     * //todo change implementation
     * only keep the first numFeatures features
     * @param clfDataSet
     * @param numFeatures
     * @return
     */
    public static ClfDataSet trim(ClfDataSet clfDataSet, int numFeatures){
        if (numFeatures> clfDataSet.getNumFeatures()){
            throw new IllegalArgumentException("numFeatures > clfDataSet.getNumFeatures()");
        }
        ClfDataSet trimmed ;
        int numClasses = clfDataSet.getNumClasses();
        boolean missingValue = clfDataSet.hasMissingValue();
        // keep density
        if (clfDataSet.isDense()) {
            trimmed = new DenseClfDataSet(clfDataSet.getNumDataPoints(), numFeatures, missingValue, numClasses);
        } else{
            trimmed = new SparseClfDataSet(clfDataSet.getNumDataPoints(),numFeatures, missingValue, numClasses);
        }
        for (int i=0;i<trimmed.getNumDataPoints();i++){
            FeatureRow featureRow = clfDataSet.getFeatureRow(i);
            //only copy non-zero elements
            Vector vector = featureRow.getVector();
            for (Vector.Element element: vector.nonZeroes()){
                int featureIndex = element.index();
                double value = element.get();
                if (featureIndex<numFeatures){
                    trimmed.setFeatureValue(i,featureIndex,value);
                }
            }
        }
        //copy labels
        int[] labels = clfDataSet.getLabels();
        for (int i=0;i<trimmed.getNumDataPoints();i++){
            trimmed.setLabel(i,labels[i]);
        }
        //just copy settings
        for (int i=0;i<trimmed.getNumDataPoints();i++){
            trimmed.getFeatureRow(i).putSetting(clfDataSet.getFeatureRow(i).getSetting().copy());
        }
        for (int j=0;j<numFeatures;j++){
            trimmed.getFeatureColumn(j).putSetting(clfDataSet.getFeatureColumn(j).getSetting().copy());
        }
        //todo double-check
        //todo: something like featuremappers
        trimmed.putSetting(clfDataSet.getSetting());
        return trimmed;
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

    public static void setLabelTranslator(ClfDataSet dataSet, LabelTranslator labelTranslator){
        int[] labels = dataSet.getLabels();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            dataSet.getFeatureRow(i).getSetting()
                    .setExtLabel(labelTranslator.toExtLabel(labels[i]));
        }
        dataSet.getSetting().setLabelTranslator(labelTranslator);
    }

//    /**
//     *
//     * @param dataSet
//     * @param extLabels in order
//     */
//    public static void setLabelTranslator(ClfDataSet dataSet, List<String> extLabels){
//        int[] labels = dataSet.getLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            dataSet.getFeatureRow(i).getSetting()
//                    .setExtLabel(extLabels.get(labels[i]));
//        }
//        dataSet.getSetting().setLabelTranslator(new LabelTranslator(extLabels));
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
//            dataSet.getFeatureRow(i).getSetting()
//                    .setExtLabel(extLabels[labels[i]]);
//        }
//        dataSet.getSetting().setLabelTranslator(new LabelTranslator(extLabels));
//    }
//
//    public static void setLabelTranslator(ClfDataSet dataSet, Map<Integer, String> intToExtLabel){
//        int[] labels = dataSet.getLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            dataSet.getFeatureRow(i).getSetting()
//                    .setExtLabel(intToExtLabel.get(labels[i]));
//        }
//        dataSet.getSetting().setLabelTranslator(new LabelTranslator(intToExtLabel));
//    }

    public static void setLabelTranslator(MultiLabelClfDataSet dataSet, LabelTranslator labelTranslator){
        MultiLabel[] multiLabels= dataSet.getMultiLabels();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            MultiLabel multiLabel = multiLabels[i];
            List<String> extLabels = multiLabel.getMatchedLabels().stream()
                    .map(labelTranslator::toExtLabel)
                    .collect(Collectors.toList());
            dataSet.getFeatureRow(i).getSetting()
                    .setExtLabels(extLabels);
        }
        dataSet.getSetting().setLabelTranslator(labelTranslator);
    }

//    public static void setLabelTranslator(MultiLabelClfDataSet dataSet, Map<Integer, String> intToExtLabel){
//        MultiLabel[] multiLabels= dataSet.getMultiLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            MultiLabel multiLabel = multiLabels[i];
//            List<String> extLabels = multiLabel.getMatchedLabels().stream()
//                    .map(intToExtLabel::get)
//                    .collect(Collectors.toList());
//            dataSet.getFeatureRow(i).getSetting()
//                    .setExtLabels(extLabels);
//        }
//        dataSet.getSetting().setLabelTranslator(new LabelTranslator(intToExtLabel));
//    }
//
//    public static void setLabelTranslator(MultiLabelClfDataSet dataSet, String[] extLabels){
//        MultiLabel[] multiLabels= dataSet.getMultiLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            MultiLabel multiLabel = multiLabels[i];
//            List<String> matchedExtLabels = multiLabel.getMatchedLabels().stream()
//                    .map(label -> extLabels[label])
//                    .collect(Collectors.toList());
//            dataSet.getFeatureRow(i).getSetting()
//                    .setExtLabels(matchedExtLabels);
//        }
//        dataSet.getSetting().setLabelTranslator(new LabelTranslator(extLabels));
//    }

    public static void setFeatureNames(DataSet dataSet, List<String> featureNames){
        if (featureNames.size()!=dataSet.getNumFeatures()){
            throw new IllegalArgumentException("featureNames.size()!=dataSet.getNumFeatures()");
        }
        for (int j=0;j<dataSet.getNumFeatures();j++){
            dataSet.getFeatureColumn(j).getSetting().setFeatureName(featureNames.get(j));
        }
    }

    public static void setFeatureNames(DataSet dataSet, String[] featureNames){
        List<String> list = Arrays.stream(featureNames).collect(Collectors.toList());
        setFeatureNames(dataSet,list);
    }

    /**
     * should use after featureMappers are finalized
     * @param dataSet
     * @param featureMappers
     */
    public static void setFeatureMappers(DataSet dataSet, FeatureMappers featureMappers){
        if (dataSet.getNumFeatures()!=featureMappers.getTotalDim()){
            throw new IllegalArgumentException("dataSet.getNumFeatures()!=featureMappers.getTotalDim()");
        }
        dataSet.getSetting().setFeatureMappers(featureMappers);
        setFeatureNames(dataSet,featureMappers.getAllNames());
    }

    /**
     * keep both local intId->extIds and global extId <-> intId translations
     * may fail because the intIds in idTranslator may not correspond to [0,numDataPoints)
     * @param dataSet
     * @param idTranslator
     */
    public static void setIdTranslator(DataSet dataSet, IdTranslator idTranslator){
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            dataSet.getFeatureRow(i).getSetting().setExtId(idTranslator.toExtId(i));
        }
        dataSet.getSetting().setIdTranslator(idTranslator);
    }

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
     * @param clfDataSet
     * @param indices
     * @return
     */
    public static ClfDataSet subSet(ClfDataSet clfDataSet, List<Integer> indices){
        ClfDataSet sample;
        int numClasses = clfDataSet.getNumClasses();
        boolean missingValue = clfDataSet.hasMissingValue();
        if (clfDataSet instanceof DenseClfDataSet){
            sample = new DenseClfDataSet(indices.size(),clfDataSet.getNumFeatures(), missingValue, numClasses);
        } else {
            sample = new SparseClfDataSet(indices.size(),clfDataSet.getNumFeatures(), missingValue, numClasses);
        }
        int[] labels = clfDataSet.getLabels();
        for (int i=0;i<indices.size();i++){
            int indexInOld = indices.get(i);
            FeatureRow oldFeatureRow = clfDataSet.getFeatureRow(indexInOld);
            int label = labels[indexInOld];
            //copy label
            sample.setLabel(i,label);
            //copy row feature values, optimized for sparse vector
            for (Vector.Element element: oldFeatureRow.getVector().nonZeroes()){
                sample.setFeatureValue(i,element.index(),element.get());
            }
            //copy data settings
            sample.getFeatureRow(i).putSetting(oldFeatureRow.getSetting());
        }

        //copy feature settings
        for (int j=0;j<clfDataSet.getNumFeatures();j++){
            sample.getFeatureColumn(j)
                    .putSetting(clfDataSet.getFeatureColumn(j).getSetting());
        }

        //safe to copy label map

        DataSetUtil.setLabelTranslator(sample, clfDataSet.getSetting().getLabelTranslator());


        //safe to copy feature mappers
        DataSetUtil.setFeatureMappers(sample,clfDataSet.getSetting().getFeatureMappers());


        //ignore idTranslator as we may have duplicate extIds
        return sample;
    }

    public static void dumpDataSettings(ClfDataSet dataSet, String file) throws IOException{
        dumpDataSettings(dataSet,new File(file));
    }

    /**
     * dump data settings to a plain text file
     * @param dataSet
     * @param file
     * @throws IOException
     */
    public static void dumpDataSettings(ClfDataSet dataSet, File file) throws IOException {
        int numDataPoints = dataSet.getNumDataPoints();
        int[] labels = dataSet.getLabels();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))
        ) {
            for (int i = 0; i < numDataPoints; i++) {
                FeatureRow featureRow = dataSet.getFeatureRow(i);
                DataSetting dataSetting = featureRow.getSetting();
                bw.write("intId=");
                bw.write("" + i);
                bw.write(",");
                bw.write("extId=");
                bw.write(dataSetting.getExtId());
                bw.write(",");
                bw.write("intLabel=");
                bw.write("" + labels[i]);
                bw.write(",");
                bw.write("extLabel=");
                bw.write(dataSetting.getExtLabel());
                bw.newLine();
            }
        }
    }

    public static void dumpDataSettings(MultiLabelClfDataSet dataSet, String file) throws IOException{
        dumpDataSettings(dataSet,new File(file));
    }

    public static void dumpDataSettings(MultiLabelClfDataSet dataSet, File file) throws IOException {
        int numDataPoints = dataSet.getNumDataPoints();
        MultiLabel[] labels = dataSet.getMultiLabels();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))
        ) {
            for (int i = 0; i < numDataPoints; i++) {
                FeatureRow featureRow = dataSet.getFeatureRow(i);
                DataSetting dataSetting = featureRow.getSetting();
                bw.write("intId=");
                bw.write("" + i);
                bw.write(",");
                bw.write("extId=");
                bw.write(dataSetting.getExtId());
                bw.write(",");
                bw.write("intLabel=");
                bw.write("" + labels[i]);
                bw.write(",");
                bw.write("extLabel=");
                bw.write(dataSetting.getExtLabels().toString());
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
    public static void dumpFeatureSettings(ClfDataSet dataSet, String file) throws IOException {
        dumpFeatureSettings(dataSet,new File(file));
    }
    public static void dumpFeatureSettings(ClfDataSet dataSet, File file) throws IOException {
        int numFeatures = dataSet.getNumFeatures();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))
        ) {
            for (int j=0;j<numFeatures;j++){
                FeatureSetting featureSetting = dataSet.getFeatureColumn(j).getSetting();
                bw.write("featureIndex=");
                bw.write(""+j);
                bw.write(",");
                bw.write("featureType=");
                if (featureSetting.getFeatureType()==FeatureType.NUMERICAL){
                    bw.write("numerical");
                } else {
                    bw.write("binary");
                }
                bw.write(",");
                bw.write("featureName=");
                bw.write(featureSetting.getFeatureName());

                bw.newLine();
            }
        }
    }


    public static void dumpFeatureSettings(MultiLabelClfDataSet dataSet, String file) throws IOException {
        dumpFeatureSettings(dataSet,new File(file));
    }

    public static void dumpFeatureSettings(MultiLabelClfDataSet dataSet, File file) throws IOException {
        int numFeatures = dataSet.getNumFeatures();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))
        ) {
            for (int j=0;j<numFeatures;j++){
                FeatureSetting featureSetting = dataSet.getFeatureColumn(j).getSetting();
                bw.write("featureIndex=");
                bw.write(""+j);
                bw.write(",");
                bw.write("featureType=");
                if (featureSetting.getFeatureType()==FeatureType.NUMERICAL){
                    bw.write("numerical");
                } else {
                    bw.write("binary");
                }
                bw.write(",");
                bw.write("featureName=");
                bw.write(featureSetting.getFeatureName());

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
            FeatureRow featureRow = dataSet.getFeatureRow(i);
            //only copy non-zero elements
            Vector vector = featureRow.getVector();
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

        for (int i=0;i<numDataPoints;i++){
            DataSetting dataSetting = dataSet.getFeatureRow(i).getSetting().copy();
            LabelTranslator labelTranslator = dataSet.getSetting().getLabelTranslator();
            int label = clfDataSet.getLabels()[i];
            if (labelTranslator!=null){
                if (label ==1){
                    dataSetting.setExtLabel(labelTranslator.toExtLabel(k));
                } else {
                    dataSetting.setExtLabel("NOT "+labelTranslator.toExtLabel(k));
                }
            } else {
                dataSetting.setExtLabel("unknown");
            }
            clfDataSet.getFeatureRow(i).putSetting(dataSetting);
        }

        for (int j=0;j<numFeatures;j++){
            FeatureSetting featureSetting = dataSet.getFeatureColumn(j).getSetting().copy();
            clfDataSet.getFeatureColumn(j).putSetting(featureSetting);
        }

        return clfDataSet;
    }


    public static void allowMissingValue(DataSet dataSet){
        if (dataSet instanceof AbstractDataSet){
            ((AbstractDataSet)dataSet).allowMissingValue();
        }
    }

}
