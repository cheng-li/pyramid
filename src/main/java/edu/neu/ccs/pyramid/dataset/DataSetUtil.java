package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;
import edu.neu.ccs.pyramid.util.SetUtil;
import edu.neu.ccs.pyramid.util.Translator;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;


import java.io.*;
import java.util.*;
import java.util.Arrays;
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
     * only keep the selected features
     * @param dataSet
     * @return
     */
    public static ClfDataSet sampleFeatures(ClfDataSet dataSet, List<Integer> columnsToKeep){
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


    public static RegDataSet sampleFeatures(RegDataSet dataSet, List<Integer> columnsToKeep){
        RegDataSet trimmed ;

        trimmed = RegDataSetBuilder.getBuilder().numDataPoints(dataSet.getNumDataPoints())
                .numFeatures(columnsToKeep.size())
                .missingValue(dataSet.hasMissingValue())
                .dense(dataSet.isDense())
                .build();

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
        double[] labels = dataSet.getLabels();
        for (int i=0;i<trimmed.getNumDataPoints();i++){
            trimmed.setLabel(i,labels[i]);
        }

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
    public static MultiLabelClfDataSet sampleFeatures(MultiLabelClfDataSet dataSet, List<Integer> columnsToKeep){
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
    public static ClfDataSet sampleFeatures(ClfDataSet clfDataSet, int numFeatures){
        List<Integer> columnsToKeep = IntStream.range(0,numFeatures).mapToObj(i->i).collect(Collectors.toList());
        return  sampleFeatures(clfDataSet, columnsToKeep);
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

        return sampleData(clfDataSet, sampledIndices);
    }

    /**
     * create a subset with the indices
     * it's fine to have duplicate indices
     * idTranslator is not saved in sampleData as we may have duplicate extIds
     * @param dataSet
     * @param indices
     * @return
     */
    public static ClfDataSet sampleData(ClfDataSet dataSet, List<Integer> indices){
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
     * create a subset with the indices
     * it's fine to have duplicate indices
     * idTranslator is not saved in sampleData as we may have duplicate extIds
     * @param dataSet
     * @param indices
     * @return
     */
    public static RegDataSet sampleData(RegDataSet dataSet, List<Integer> indices){
        RegDataSet sample;
        sample = RegDataSetBuilder.getBuilder().numDataPoints(indices.size())
                .numFeatures(dataSet.getNumFeatures())
                .missingValue(dataSet.hasMissingValue())
                .dense(dataSet.isDense())
                .build();
        double[] labels = dataSet.getLabels();
        for (int i=0;i<indices.size();i++){
            int indexInOld = indices.get(i);
            Vector oldVector = dataSet.getRow(indexInOld);
            double label = labels[indexInOld];
            //copy label
            sample.setLabel(i,label);
            //copy row feature values, optimized for sparse vector
            for (Vector.Element element: oldVector.nonZeroes()){
                sample.setFeatureValue(i,element.index(),element.get());
            }
        }
        sample.setFeatureList(dataSet.getFeatureList());

        //ignore idTranslator as we may have duplicate extIds
        return sample;
    }


    public static Pair<DataSet, double[][]> sampleData(DataSet dataSet, double[][] targetDistribution, List<Integer> indices){
        DataSet sample;
        int numClasses = targetDistribution[0].length;
        double[][] sampledTargets = new double[indices.size()][numClasses];
        sample = DataSetBuilder.getBuilder().dense(dataSet.isDense()).missingValue(dataSet.hasMissingValue())
                .numDataPoints(indices.size()).numFeatures(dataSet.getNumFeatures()).build();

        for (int i=0;i<indices.size();i++){
            int indexInOld = indices.get(i);
            Vector oldVector = dataSet.getRow(indexInOld);
            double[] targets = targetDistribution[indexInOld];
            //copy label
            sampledTargets[i] = Arrays.copyOf(targets,targets.length);
            //copy row feature values, optimized for sparse vector
            for (Vector.Element element: oldVector.nonZeroes()){
                sample.setFeatureValue(i,element.index(),element.get());
            }

        }

        sample.setFeatureList(dataSet.getFeatureList());

        //ignore idTranslator as we may have duplicate extIds
        return new Pair<>(sample, sampledTargets);
    }

    /**
     * create a subset with the indices
     * it's fine to have duplicate indices
     * @param dataSet
     * @param indices
     * @return
     */
    public static MultiLabelClfDataSet sampleData(MultiLabelClfDataSet dataSet, List<Integer> indices){
        MultiLabelClfDataSet sample;
        sample = MLClfDataSetBuilder.getBuilder()
                .numClasses(dataSet.getNumClasses())
                .numDataPoints(indices.size())
                .numFeatures(dataSet.getNumFeatures())
                .missingValue(dataSet.hasMissingValue())
                .density(dataSet.density())
                .build();
        MultiLabel[] labels = dataSet.getMultiLabels();
        IdTranslator idTranslator = new IdTranslator();
        for (int i=0;i<indices.size();i++){
            int indexInOld = indices.get(i);
            String extId = dataSet.getIdTranslator().toExtId(indexInOld);
            idTranslator.addData(i, extId);
            Vector oldVector = dataSet.getRow(indexInOld);
            Set<Integer> label = labels[indexInOld].getMatchedLabels();
            //copy label
            sample.addLabels(i,label);
            //copy row feature values, optimized for sparse vector
            for (Vector.Element element: oldVector.nonZeroes()){
                sample.setFeatureValue(i,element.index(),element.get());
            }
        }
        sample.setFeatureList(dataSet.getFeatureList());
        sample.setIdTranslator(idTranslator);
        sample.setLabelTranslator(dataSet.getLabelTranslator());
        return sample;
    }


    /**
     * assuming they have different feature sets
     * @param dataSet1
     * @param dataSet2
     * @return
     */
    public static ClfDataSet concatenateByColumn(ClfDataSet dataSet1, ClfDataSet dataSet2){
        int numDataPoints = dataSet1.getNumDataPoints();
        int numFeatures1 = dataSet1.getNumFeatures();
        int numFeatures2 = dataSet2.getNumFeatures();
        int numFeatures = numFeatures1 + numFeatures2;
        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(dataSet1.getNumClasses())
                .dense(dataSet1.isDense())
                .missingValue(dataSet1.hasMissingValue())
                .build();

        int featureIndex = 0;
        for (int j=0;j<numFeatures1;j++){
            Vector vector = dataSet1.getColumn(j);
            for (Vector.Element element: vector.nonZeroes()){
                int i = element.index();
                double value = element.get();
                dataSet.setFeatureValue(i,featureIndex,value);
            }
            featureIndex += 1;
        }


        for (int j=0;j<numFeatures2;j++){
            Vector vector = dataSet2.getColumn(j);
            for (Vector.Element element: vector.nonZeroes()){
                int i = element.index();
                double value = element.get();
                dataSet.setFeatureValue(i,featureIndex,value);
            }
            featureIndex += 1;
        }

        int[] labels = dataSet1.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.setLabel(i,labels[i]);
        }

        FeatureList featureList = new FeatureList();
        for (Feature feature: dataSet1.getFeatureList().getAll()){
            featureList.add(feature);
        }

        for (Feature feature: dataSet2.getFeatureList().getAll()){
            featureList.add(feature);
        }

        dataSet.setFeatureList(featureList);

        dataSet.setLabelTranslator(dataSet1.getLabelTranslator());
        dataSet.setIdTranslator(dataSet1.getIdTranslator());

        return dataSet;
    }


    public static ClfDataSet concatenateByRow(ClfDataSet dataSet1, ClfDataSet dataSet2){
        int numDataPoints1 = dataSet1.getNumDataPoints();
        int numDataPoints2 = dataSet2.getNumDataPoints();
        int numDataPoints = numDataPoints1 + numDataPoints2;
        int numFeatures = dataSet1.getNumFeatures();

        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(dataSet1.getNumClasses())
                .dense(dataSet1.isDense())
                .missingValue(dataSet1.hasMissingValue())
                .build();


        int dataIndex = 0;
        for (int i=0;i<dataSet1.getNumDataPoints();i++){
            Vector row = dataSet1.getRow(i);
            for (Vector.Element element: row.nonZeroes()){
                int j = element.index();
                double value = element.get();
                dataSet.setFeatureValue(dataIndex,j,value);
            }
            dataSet.setLabel(dataIndex,dataSet1.getLabels()[i]);
            dataIndex+=1;
        }

        for (int i=0;i<dataSet2.getNumDataPoints();i++){
            Vector row = dataSet2.getRow(i);
            for (Vector.Element element: row.nonZeroes()){
                int j = element.index();
                double value = element.get();
                dataSet.setFeatureValue(dataIndex,j,value);
            }
            dataSet.setLabel(dataIndex,dataSet2.getLabels()[i]);
            dataIndex+=1;
        }

        dataSet.setFeatureList(dataSet1.getFeatureList());
        dataSet.setLabelTranslator(dataSet1.getLabelTranslator());
        //id translator is not set

        return dataSet;
    }

    public static MultiLabelClfDataSet concatenateByColumn(MultiLabelClfDataSet dataSet1, MultiLabelClfDataSet dataSet2){
        int numDataPoints = dataSet1.getNumDataPoints();
        int numFeatures1 = dataSet1.getNumFeatures();
        int numFeatures2 = dataSet2.getNumFeatures();
        int numFeatures = numFeatures1 + numFeatures2;
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(dataSet1.getNumClasses())
                .dense(dataSet1.isDense())
                .missingValue(dataSet1.hasMissingValue())
                .build();

        int featureIndex = 0;
        for (int j=0;j<numFeatures1;j++){
            Vector vector = dataSet1.getColumn(j);
            for (Vector.Element element: vector.nonZeroes()){
                int i = element.index();
                double value = element.get();
                dataSet.setFeatureValue(i,featureIndex,value);
            }
            featureIndex += 1;
        }


        for (int j=0;j<numFeatures2;j++){
            Vector vector = dataSet2.getColumn(j);
            for (Vector.Element element: vector.nonZeroes()){
                int i = element.index();
                double value = element.get();
                dataSet.setFeatureValue(i,featureIndex,value);
            }
            featureIndex += 1;
        }

        MultiLabel[] labels = dataSet1.getMultiLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.setLabels(i,labels[i]);
        }

        FeatureList featureList = new FeatureList();
        for (Feature feature: dataSet1.getFeatureList().getAll()){
            featureList.add(feature);
        }

        for (Feature feature: dataSet2.getFeatureList().getAll()){
            featureList.add(feature);
        }

        dataSet.setFeatureList(featureList);

        dataSet.setLabelTranslator(dataSet1.getLabelTranslator());
        dataSet.setIdTranslator(dataSet1.getIdTranslator());

        return dataSet;
    }

    /**
     *
     * @param dataSet
     * @param numFolds
     * @param foldIndices 1 - numfolds
     * @return
     */
    public static ClfDataSet sampleByFold(ClfDataSet dataSet, int numFolds, Set<Integer> foldIndices){
        for (int fold: foldIndices){
            boolean con = fold>=1 && fold<=numFolds;
            if (! con){
                throw new IllegalArgumentException("should have fold>=1 && fold<=numFolds");
            }
        }

        int numData = dataSet.getNumDataPoints();
        List<Integer> keep = new ArrayList<>();
        for (int i=0;i<numData;i++){
            int rem = i%numFolds;
            if (foldIndices.contains(rem+1)){
                keep.add(i);
            }
        }

        return sampleData(dataSet,keep);
    }


    public static List<ClfDataSet> partitionToBatches(ClfDataSet dataSet, int numBatches){
        List<ClfDataSet> batches = new ArrayList<>();
        for (int i=1;i<=numBatches;i++){
            Set<Integer> index = new HashSet<>();
            index.add(i);
            batches.add(sampleByFold(dataSet, numBatches, index));
        }
        return batches;
    }

    public static MultiLabelClfDataSet sampleByFold(MultiLabelClfDataSet dataSet, int numFolds, Set<Integer> foldIndices){
        for (int fold: foldIndices){
            boolean con = fold>=1 && fold<=numFolds;
            if (! con){
                throw new IllegalArgumentException("should have fold>=1 && fold<=numFolds");
            }
        }

        int numData = dataSet.getNumDataPoints();
        List<Integer> keep = new ArrayList<>();
        for (int i=0;i<numData;i++){
            int rem = i%numFolds;
            if (foldIndices.contains(rem+1)){
                keep.add(i);
            }
        }

        return sampleData(dataSet,keep);
    }

    public static List<MultiLabelClfDataSet> partitionToBatches(MultiLabelClfDataSet dataSet, int numBatches){
        List<MultiLabelClfDataSet> batches = new ArrayList<>();
        for (int i=1;i<=numBatches;i++){
            Set<Integer> index = new HashSet<>();
            index.add(i);
            batches.add(sampleByFold(dataSet, numBatches, index));
        }
        return batches;
    }


    /**
     *
     * @param dataSet
     * @param numFolds
     * @param foldIndices 1 - numfolds
     * @return
     */
    public static RegDataSet sampleByFold(RegDataSet dataSet, int numFolds, Set<Integer> foldIndices){
        for (int fold: foldIndices){
            boolean con = fold>=1 && fold<=numFolds;
            if (! con){
                throw new IllegalArgumentException("should have fold>=1 && fold<=numFolds");
            }
        }

        int numData = dataSet.getNumDataPoints();
        List<Integer> keep = new ArrayList<>();
        for (int i=0;i<numData;i++){
            int rem = i%numFolds;
            if (foldIndices.contains(rem+1)){
                keep.add(i);
            }
        }

        return sampleData(dataSet,keep);
    }

    public static Pair<DataSet,double[][]> sampleByFold(DataSet dataSet, double[][] targetDistribution, int numFolds, Set<Integer> foldIndices){
        for (int fold: foldIndices){
            boolean con = fold>=1 && fold<=numFolds;
            if (! con){
                throw new IllegalArgumentException("should have fold>=1 && fold<=numFolds");
            }
        }

        int numData = dataSet.getNumDataPoints();
        List<Integer> keep = new ArrayList<>();
        for (int i=0;i<numData;i++){
            int rem = i%numFolds;
            if (foldIndices.contains(rem+1)){
                keep.add(i);
            }
        }

        return sampleData(dataSet,targetDistribution, keep);
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
        pair.setFirst(DataSetUtil.sampleData(clfDataSet, trainIndices));
        pair.setSecond(DataSetUtil.sampleData(clfDataSet, testIndices));
        return pair;
    }


    /**
     *
     * @param dataSet
     * @return training set and validation set
     */
    public static Pair<RegDataSet,RegDataSet> splitToTrainValidation(RegDataSet dataSet, double trainPercentage){
        int numDataPoints = dataSet.getNumDataPoints();
        List<Integer> all = IntStream.range(0,dataSet.getNumDataPoints()).mapToObj(i->i).collect(Collectors.toList());
        List<Integer> trainIndices = Sampling.sampleByPercentage(all,trainPercentage);

        Set<Integer> testIndicesSet = new HashSet<>();
        for (int i=0;i<numDataPoints;i++){
            testIndicesSet.add(i);
        }
        testIndicesSet.removeAll(trainIndices);
        List<Integer> testIndices = testIndicesSet.stream().collect(Collectors.toList());
        Pair<RegDataSet,RegDataSet> pair = new Pair<>();
        pair.setFirst(DataSetUtil.sampleData(dataSet, trainIndices));
        pair.setSecond(DataSetUtil.sampleData(dataSet, testIndices));
        return pair;
    }

    /**
     *
     * @param multiLabelClfDataSet
     * @param trainPercentage
     * @return
     */
    public static Pair<MultiLabelClfDataSet, MultiLabelClfDataSet> splitToTrainValidation(MultiLabelClfDataSet multiLabelClfDataSet,
                                                                                          double trainPercentage) {
        int numDataPoints = multiLabelClfDataSet.getNumDataPoints();
        List<Integer> all = IntStream.range(0,numDataPoints).mapToObj(i->i).collect(Collectors.toList());
        List<Integer> trainIndices = Sampling.sampleByPercentage(all, trainPercentage);

        Set<Integer> testIndicesSet = new HashSet<>();
        for (int i=0; i<numDataPoints; i++) {
            testIndicesSet.add(i);
        }
        testIndicesSet.removeAll(trainIndices);
        List<Integer> testIndices = testIndicesSet.stream().collect(Collectors.toList());
        Pair<MultiLabelClfDataSet, MultiLabelClfDataSet> pair = new Pair<>();
        pair.setFirst(DataSetUtil.sampleData(multiLabelClfDataSet, trainIndices));
        pair.setSecond(DataSetUtil.sampleData(multiLabelClfDataSet, testIndices));
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


    public static List<MultiLabel> gatherMultiLabels(MultiLabelClfDataSet dataSet){
        Set<MultiLabel> multiLabels = new HashSet<>();
        MultiLabel[] multiLabelsArray = dataSet.getMultiLabels();
        for (MultiLabel multiLabel: multiLabelsArray){
            multiLabels.add(multiLabel);
        }
        return multiLabels.stream().collect(Collectors.toList());
    }

    public static Set<Integer> gatherLabels(MultiLabelClfDataSet dataSet){
        Set<Integer> labels = new HashSet<>();
        MultiLabel[] multiLabelsArray = dataSet.getMultiLabels();
        for (MultiLabel multiLabel: multiLabelsArray){
            labels.addAll(multiLabel.getMatchedLabels());
        }
        return labels;
    }

    public static int[] toBinaryLabels(MultiLabel[] multiLabels, int k){
        int[] binaryLabels = new int[multiLabels.length];
        for (int i=0;i<multiLabels.length;i++){
            if (multiLabels[i].matchClass(k)){
                binaryLabels[i]=1;
            }
        }
        return binaryLabels;
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


    public static Pair<ClfDataSet,Translator<MultiLabel>> toMultiClass(MultiLabelClfDataSet dataSet){
        int numDataPoints = dataSet.getNumDataPoints();
        int numFeatures = dataSet.getNumFeatures();
        List<MultiLabel> multiLabels = DataSetUtil.gatherMultiLabels(dataSet);
        Translator<MultiLabel> translator = new Translator<>();
        translator.addAll(multiLabels);
        ClfDataSet clfDataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(numFeatures)
                .dense(dataSet.isDense())
                .missingValue(dataSet.hasMissingValue())
                .numClasses(translator.size())
                .build();

        for (int i=0;i<numDataPoints;i++){
            //only copy non-zero elements
            Vector vector = dataSet.getRow(i);
            for (Vector.Element element: vector.nonZeroes()){
                int featureIndex = element.index();
                double value = element.get();
                clfDataSet.setFeatureValue(i,featureIndex,value);
            }
            int label = translator.getIndex(dataSet.getMultiLabels()[i]);
            clfDataSet.setLabel(i,label);
        }

        List<String> extLabels = multiLabels.stream().map(MultiLabel::toString).collect(Collectors.toList());
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);
        clfDataSet.setLabelTranslator(labelTranslator);
        clfDataSet.setFeatureList(dataSet.getFeatureList());
        return new Pair<>(clfDataSet,translator);
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

    public static void setFeatureNames(DataSet dataSet, List<String> names){
        if (dataSet.getNumFeatures()!=names.size()){
            throw new IllegalArgumentException("dataSet.getNumFeatures()!=names.size()");
        }

        for (int i=0;i<names.size();i++){
            dataSet.getFeatureList().get(i).setName(names.get(i));
        }
    }

    /**
     * make every non-zero feature 1
     * @param dataSet
     */
    public static void binarizeFeature(DataSet dataSet){
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            List<Integer> nonZeors = new ArrayList<>();
            Vector row = dataSet.getRow(i);
            for (Vector.Element element: row.nonZeroes()){
                nonZeors.add(element.index());
            }
            for (int j:nonZeors){
                dataSet.setFeatureValue(i,j,1);
            }
        }
    }

    public static double[][] labelDistribution(ClfDataSet dataSet){
        int numData = dataSet.getNumDataPoints();
        int numClass = dataSet.getNumClasses();
        double[][] dis = new double[numData][numClass];
        int[] labels = dataSet.getLabels();
        for (int i=0;i<numData;i++){
            int label = labels[i];
            dis[i][label] = 1.0;
        }
        return dis;
    }

    public static String multiLabelToBinaryString(MultiLabelClfDataSet dataSet){
        int numData = dataSet.getNumDataPoints();
        int numClasses = dataSet.getNumClasses();
        StringBuilder sb = new StringBuilder();
        for (int i=0;i<numData;i++){
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            for (int l=0;l<numClasses;l++){
                String bit;
                if (multiLabel.matchClass(l)){
                    bit = "1";
                } else {
                    bit = "0";
                }
                sb.append(bit);
                if (l<numClasses-1){
                    sb.append(" ");
                }
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public static void detectDuplicate(MultiLabelClfDataSet train, MultiLabelClfDataSet test){
        Set<Vector> vectors = new HashSet<>();
        for (int i=0;i<train.getNumDataPoints();i++){
            vectors.add(train.getRow(i));
        }

        List<Integer> duplicate = new ArrayList<>();
        for (int i=0;i<test.getNumDataPoints();i++){
            if (vectors.contains(test.getRow(i))){
                duplicate.add(i);
            }
        }
        System.out.println("number of test data points which occur in training set = "+duplicate.size());
        System.out.println("duplicates = "+duplicate);
    }

    public static void dataComparasion(MultiLabelClfDataSet trainSet, MultiLabelClfDataSet testSet){
        System.out.println("---------------------------------Data Comparasion------------------------------");
        System.out.println("Number of Features: " + trainSet.getNumFeatures());
        System.out.println("Number of Labels: " + trainSet.getNumClasses());
        System.out.println("Number of Training: " + trainSet.getNumDataPoints());
        System.out.println("Number of Testing: " + testSet.getNumDataPoints());

        Set<MultiLabel> trainLabelSet = new HashSet<>();
        Set<MultiLabel> testLabelSet  = new HashSet<>();

        for (MultiLabel multiLabel : trainSet.getMultiLabels()) {
            trainLabelSet.add(multiLabel);
        }
        for (MultiLabel multiLabel : testSet.getMultiLabels()) {
            testLabelSet.add(multiLabel);
        }


        System.out.println("Train label Cardinality: " + trainSet.labelCardinality());
        System.out.println("Test label Cardinality: " + testSet.labelCardinality());
        System.out.println("Train label Density: " + trainSet.labelDensity());
        System.out.println("Test label Density: " + testSet.labelDensity());
        System.out.println();

        System.out.println("Train distinct label num: " + trainLabelSet.size());
        System.out.println("Test distinct label num: " + testLabelSet.size());
        Set<MultiLabel> unionSet = SetUtil.union(trainLabelSet, testLabelSet);
        System.out.println("Union distinct label num: " + unionSet.size());
        Set<MultiLabel> intersectSet = SetUtil.intersect(trainLabelSet, testLabelSet);
        System.out.println("Intersect distinct label num: " + intersectSet.size());

        Set<MultiLabel> newTestSet = SetUtil.complement(testLabelSet, trainLabelSet);
        System.out.println("New label combination number in test: " + newTestSet.size());

        int newTestLabelCounts = 0;
        for (MultiLabel label : testSet.getMultiLabels()) {
            if (newTestSet.contains(label)) {
                newTestLabelCounts++;
            }
        }
        System.out.println("New label combination data counts: " + newTestLabelCounts);
        System.out.println("New label combination data rate: " + (double)newTestLabelCounts/testSet.getNumDataPoints());
        System.out.println("---------------------------------------------------------------");
        System.out.println();
        System.out.println();
    }
}
