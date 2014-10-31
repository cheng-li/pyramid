package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;
import org.apache.mahout.math.Vector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by chengli on 10/28/14.
 */
public class ArffFormat {

    private static final String ARFF_MATRIX_FILE_NAME = "feature_matrix.arff";
    private static final String ARFF_CONFIG_FILE_NAME = "config.txt";
    private static final String ARFF_CONFIG_NUM_DATA_POINTS = "numDataPoints";
    private static final String ARFF_CONFIG_NUM_FEATURES = "numFeatures";
    private static final String ARFF_CONFIG_NUM_CLASSES = "numClasses";

    public static void save(ClfDataSet dataSet, String arffFile){
        save(dataSet,new File(arffFile));
    }

    public static void save(RegDataSet dataSet, String arffFile){
        save(dataSet,new File(arffFile));
    }

    public static void save(MultiLabelClfDataSet dataSet, String arffFile){
        save(dataSet,new File(arffFile));
    }

    public static void save(ClfDataSet dataSet, File arffFile){
        if (!arffFile.exists()){
            arffFile.mkdirs();
        }
        writeMatrixFile(dataSet, arffFile);
        writeConfigFile(dataSet, arffFile);
    }

    public static void save(RegDataSet dataSet, File arffFile){
        if (!arffFile.exists()){
            arffFile.mkdirs();
        }
        writeMatrixFile(dataSet, arffFile);
        writeConfigFile(dataSet, arffFile);
    }

    public static void save(MultiLabelClfDataSet dataSet, File arffFile){
        if (!arffFile.exists()){
            arffFile.mkdirs();
        }
        writeMatrixFile(dataSet, arffFile);
        writeConfigFile(dataSet, arffFile);
    }

    private static void writeMatrixFile(ClfDataSet dataSet, File arffFile) {
        File matrixFile = new File(arffFile, ARFF_MATRIX_FILE_NAME);
        int numDataPoints = dataSet.getNumDataPoints();
        int numFeatures = dataSet.getNumFeatures();
        int[] labels = dataSet.getLabels();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(matrixFile));
        ) {
            bw.write("@RELATION MATRIX" + "\n");
            for (int i=0; i<numFeatures; i++){
                bw.write("@ATTRIBUTE " + i + " NUMERIC" + "\n");
            }
            bw.write("@ATTRIBUTE class {0");
            for (int i=1; i<dataSet.getNumClasses(); i++){
                bw.write("," + i);
            }
            bw.write("}" + "\n");
            bw.write("@DATA" + "\n");
            for (int i=0; i<numDataPoints; i++){
                int label = labels[i];
                bw.write("{");
                FeatureRow featureRow = dataSet.getFeatureRow(i);
                Vector vector = featureRow.getVector();
                // only write non-zeros
                for (Vector.Element element: vector.nonZeroes()){
                    int featureIndex = element.index();
                    double featureValue = element.get();
                    bw.write(featureIndex+" "+featureValue+",");
                }
                bw.write(numFeatures+" "+label+"}"+"\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void writeMatrixFile(RegDataSet dataSet, File arffFile) {
        File matrixFile = new File(arffFile, ARFF_MATRIX_FILE_NAME);
        int numDataPoints = dataSet.getNumDataPoints();
        int numFeatures = dataSet.getNumFeatures();
        double[] labels = dataSet.getLabels();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(matrixFile));
        ) {
            bw.write("@RELATION MATRIX" + "\n");
            for (int i=0; i<numFeatures; i++){
                bw.write("@ATTRIBUTE " + i + " NUMERIC" + "\n");
            }
            bw.write("@ATTRIBUTE class NUMERIC" + "\n");
            bw.write("@DATA" + "\n");
            for (int i=0; i<numDataPoints; i++){
                double label = labels[i];
                bw.write("{");
                FeatureRow featureRow = dataSet.getFeatureRow(i);
                Vector vector = featureRow.getVector();
                // only write non-zeros
                for (Vector.Element element: vector.nonZeroes()){
                    int featureIndex = element.index();
                    double featureValue = element.get();
                    bw.write(featureIndex+" "+featureValue+",");
                }
                bw.write(numFeatures+" "+label+"}"+"\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void writeMatrixFile(MultiLabelClfDataSet dataSet, File arffFile) {
        File matrixFile = new File(arffFile, ARFF_MATRIX_FILE_NAME);
        int numDataPoints = dataSet.getNumDataPoints();
        int numFeatures = dataSet.getNumFeatures();
        MultiLabel[] multiLabels = dataSet.getMultiLabels();
        Set<Integer> allLabels = unionLabels(multiLabels);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(matrixFile));
        ) {
            bw.write("@RELATION MATRIX" + "\n");
            for (int i=0; i<numFeatures; i++){
                bw.write("@ATTRIBUTE " + i + " NUMERIC" + "\n");
            }
            for (int i=0; i<allLabels.size(); i++){
                bw.write("@ATTRIBUTE class " + i + " {0,1}" + "\n");
            }
            bw.write("@DATA" + "\n");
            for (int i=0; i<numDataPoints; i++){
                MultiLabel multiLabel = multiLabels[i];
                List<Integer> labels = multiLabel.getMatchedLabels().stream().sorted().collect(Collectors.toList());
                bw.write("{");
                FeatureRow featureRow = dataSet.getFeatureRow(i);
                Vector vector = featureRow.getVector();
                // only write non-zeros
                for (Vector.Element element: vector.nonZeroes()){
                    int featureIndex = element.index();
                    double featureValue = element.get();
                    bw.write(featureIndex+" "+featureValue+",");
                }
                for (int l=0;l<labels.size()-1;l++){
                    int label = labels.get(l) + numFeatures;
                    bw.write(label+" 1,");
                }
                int label = labels.get(labels.size()-1) + numFeatures;
                bw.write(label+" 1}"+"\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Set<Integer> unionLabels(MultiLabel[] multiLabels) {
        Set<Integer> labels = new HashSet<>();
        for (int i=0; i<multiLabels.length; i++){
            labels.addAll(multiLabels[i].getMatchedLabels());
        }
        return labels;
    }

    private static void writeConfigFile(ClfDataSet dataSet, File arffFile) {
        File configFile = new File(arffFile, ARFF_CONFIG_FILE_NAME);
        Config config = new Config();
        config.setInt(ARFF_CONFIG_NUM_DATA_POINTS,dataSet.getNumDataPoints());
        config.setInt(ARFF_CONFIG_NUM_FEATURES,dataSet.getNumFeatures());
        config.setInt(ARFF_CONFIG_NUM_CLASSES,dataSet.getNumClasses());
        try {
            config.store(configFile);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void writeConfigFile(RegDataSet dataSet, File arffFile) {
        File configFile = new File(arffFile, ARFF_CONFIG_FILE_NAME);
        Config config = new Config();
        config.setInt(ARFF_CONFIG_NUM_DATA_POINTS,dataSet.getNumDataPoints());
        config.setInt(ARFF_CONFIG_NUM_FEATURES,dataSet.getNumFeatures());
        try {
            config.store(configFile);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void writeConfigFile(MultiLabelClfDataSet dataSet, File arffFile) {
        File configFile = new File(arffFile, ARFF_CONFIG_FILE_NAME);
        Config config = new Config();
        config.setInt(ARFF_CONFIG_NUM_DATA_POINTS,dataSet.getNumDataPoints());
        config.setInt(ARFF_CONFIG_NUM_FEATURES,dataSet.getNumFeatures());
        config.setInt(ARFF_CONFIG_NUM_CLASSES,dataSet.getNumClasses());
        try {
            config.store(configFile);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static ClfDataSet loadClfDataSet(String arffFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }

    public static RegDataSet loadRegDataSet(String arffFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }

    public static MultiLabelClfDataSet loaMultiLabelClfDataSet(String arffFile, DataSetType dataSetType,
                                                               boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }


}
