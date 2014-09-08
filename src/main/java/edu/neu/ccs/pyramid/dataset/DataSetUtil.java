package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.feature.FeatureMappers;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by chengli on 8/7/14.
 */
public class DataSetUtil {

    /**
     * only keep the first numFeatures features
     * @param clfDataSet
     * @param numFeatures
     * @return
     */
    public static ClfDataSet trim(ClfDataSet clfDataSet, int numFeatures){
        if (numFeatures> clfDataSet.getNumFeatures()){
            throw new IllegalArgumentException("numFeatures> clfDataSet.getNumFeatures()");
        }
        ClfDataSet trimmed ;
        // keep density
        if (clfDataSet.isDense()) {
            trimmed = new DenseClfDataSet(clfDataSet.getNumDataPoints(), numFeatures);
        } else{
            trimmed = new SparseClfDataSet(clfDataSet.getNumDataPoints(),numFeatures);
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
            trimmed.getFeatureRow(i).putSetting(clfDataSet.getFeatureRow(i).getSetting());
        }
        for (int j=0;j<numFeatures;j++){
            trimmed.getFeatureColumn(j).putSetting(clfDataSet.getFeatureColumn(j).getSetting());
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

    /**
     *
     * @param dataSet
     * @param extLabels in order
     */
    public static void setExtLabels(ClfDataSet dataSet, List<String> extLabels){
        int[] labels = dataSet.getLabels();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            dataSet.getFeatureRow(i).getSetting()
                    .setExtLabel(extLabels.get(labels[i]));
        }
        dataSet.getSetting().setLabelMap(extLabels.toArray(new String[extLabels.size()]));
    }

    /**
     *
     * @param dataSet
     * @param extLabels in order
     */
    public static void setExtLabels(ClfDataSet dataSet, String[] extLabels){
        int[] labels = dataSet.getLabels();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            dataSet.getFeatureRow(i).getSetting()
                    .setExtLabel(extLabels[labels[i]]);
        }
        dataSet.getSetting().setLabelMap(extLabels);
    }

    public static void setExtLabels(ClfDataSet dataSet, Map<Integer,String> intToExtLabel){
        int[] labels = dataSet.getLabels();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            dataSet.getFeatureRow(i).getSetting()
                    .setExtLabel(intToExtLabel.get(labels[i]));
        }
        dataSet.getSetting().setLabelMap(intToExtLabel);
    }

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

}
