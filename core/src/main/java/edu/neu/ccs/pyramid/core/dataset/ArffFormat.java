package edu.neu.ccs.pyramid.core.dataset;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.util.Pair;
import org.apache.mahout.math.Vector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chengli on 10/28/14.
 */
public class ArffFormat {

    //todo check missing values
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
                Vector vector = dataSet.getRow(i);
                // only write non-zeros
                // only write non-zeros
                List<Pair<Integer,Double>> pairs = new ArrayList<>();
                for (Vector.Element element:vector.nonZeroes()){
                    Pair<Integer,Double> pair = new Pair<>(element.index(),element.get());
                    pairs.add(pair);
                }
                Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getFirst);
                List<Pair<Integer,Double>> sorted = pairs.stream().sorted(comparator)
                        .collect(Collectors.toList());
                for (Pair<Integer,Double> pair: sorted){
                    bw.write(pair.getFirst()+":"+pair.getSecond()+" ");
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
                Vector vector = dataSet.getRow(i);
                // only write non-zeros
                List<Pair<Integer,Double>> pairs = new ArrayList<>();
                for (Vector.Element element:vector.nonZeroes()){
                    Pair<Integer,Double> pair = new Pair<>(element.index(),element.get());
                    pairs.add(pair);
                }
                Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getFirst);
                List<Pair<Integer,Double>> sorted = pairs.stream().sorted(comparator)
                        .collect(Collectors.toList());
                for (Pair<Integer,Double> pair: sorted){
                    bw.write(pair.getFirst()+":"+pair.getSecond()+" ");
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
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(matrixFile));
        ) {
            bw.write("@RELATION MATRIX" + "\n");
            for (int i=0; i<numFeatures; i++){
                bw.write("@ATTRIBUTE " + i + " NUMERIC" + "\n");
            }
            for (int i=0; i<dataSet.getNumClasses(); i++){
                bw.write("@ATTRIBUTE class " + i + " {0,1}" + "\n");
            }
            bw.write("@DATA" + "\n");
            for (int i=0; i<numDataPoints; i++){
                MultiLabel multiLabel = multiLabels[i];
                List<Integer> labels = multiLabel.getMatchedLabels().stream().sorted().collect(Collectors.toList());
                bw.write("{");
                Vector vector = dataSet.getRow(i);
                // only write non-zeros
                List<Pair<Integer,Double>> pairs = new ArrayList<>();
                for (Vector.Element element:vector.nonZeroes()){
                    Pair<Integer,Double> pair = new Pair<>(element.index(),element.get());
                    pairs.add(pair);
                }
                Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getFirst);
                List<Pair<Integer,Double>> sorted = pairs.stream().sorted(comparator)
                        .collect(Collectors.toList());
                for (Pair<Integer,Double> pair: sorted){
                    bw.write(pair.getFirst()+":"+pair.getSecond()+" ");
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
