package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/19/14.
 */
public class TRECFormat {
    /**
     * internally, trecFile is a directory with 4 files in it
     * one for sparse feature matrix, one for config,
     * one for data settings, one for feature settings
     */
    private static final String TREC_MATRIX_FILE_NAME = "feature_matrix.txt";
    private static final String TREC_CONFIG_FILE_NAME = "config.txt";
    private static final String TREC_CONFIG_NUM_DATA_POINTS = "numDataPoints";
    private static final String TREC_CONFIG_NUM_FEATURES = "numFeatures";
    private static final String TREC_DATA_SETTINGS_FILE_NAME = "data_settings.ser";
    private static final String TREC_FEATURE_SETTINGS_FILE_NAME = "feature_settings.ser";

    public static void save(ClfDataSet dataSet, String trecFile){
        save(dataSet,new File(trecFile));
    }

    public static void save(RegDataSet dataSet, String trecFile){
        save(dataSet,new File(trecFile));
    }

    public static void save(ClfDataSet dataSet, File trecFile){
        if (!trecFile.exists()){
            trecFile.mkdirs();
        }
        writeMatrixFile(dataSet, trecFile);
        writeConfigFile(dataSet, trecFile);
        writeDataSettings(dataSet,trecFile);
        writeFeatureSettings(dataSet,trecFile);
    }

    public static void save(RegDataSet dataSet, File trecFile) {
        if (!trecFile.exists()){
            trecFile.mkdirs();
        }
        writeMatrixFile(dataSet, trecFile);
        writeConfigFile(dataSet, trecFile);
        writeDataSettings(dataSet,trecFile);
        writeFeatureSettings(dataSet,trecFile);

    }

    public static ClfDataSet loadClfDataSet(String trecFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return loadClfDataSet(new File(trecFile),dataSetType, loadSettings);
    }

    public static RegDataSet loadRegDataSet(String trecFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return loadRegDataSet(new File(trecFile),dataSetType, loadSettings);
    }


    public static ClfDataSet loadClfDataSet(File trecFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        boolean legalArg = ((dataSetType == DataSetType.CLF_DENSE)
                ||(dataSetType==DataSetType.CLF_SPARSE));
        if (!legalArg){
            throw new IllegalArgumentException("illegal data set type");
        }
        int numDataPoints = parseNumDataPoints(trecFile);
        int numFeatures = parseNumFeaturess(trecFile);
        ClfDataSet clfDataSet = null;
        if (dataSetType==DataSetType.CLF_DENSE){
            clfDataSet = new DenseClfDataSet(numDataPoints,numFeatures);
        }
        if (dataSetType==DataSetType.CLF_SPARSE){
            clfDataSet = new SparseClfDataSet(numDataPoints,numFeatures);
        }
        fillClfDataSet(clfDataSet,trecFile);
        if (loadSettings){
            loadDataSettings(clfDataSet,trecFile);
            loadFeatureSettings(clfDataSet,trecFile);
        }

        return clfDataSet;
    }

    public static RegDataSet loadRegDataSet(File trecFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        boolean legalArg = ((dataSetType == DataSetType.REG_DENSE)
                ||(dataSetType==DataSetType.REG_SPARSE));
        if (!legalArg){
            throw new IllegalArgumentException("illegal data set type");
        }
        int numDataPoints = parseNumDataPoints(trecFile);
        int numFeatures = parseNumFeaturess(trecFile);
        RegDataSet regDataSet = null;
        if (dataSetType==DataSetType.REG_DENSE){
            regDataSet = new DenseRegDataSet(numDataPoints,numFeatures);
        }
        if (dataSetType==DataSetType.REG_SPARSE){
            throw new RuntimeException("currently not supported ");
        }
        fillRegDataSet(regDataSet, trecFile);
        if (loadSettings){
            loadDataSettings(regDataSet,trecFile);
            loadFeatureSettings(regDataSet,trecFile);
        }

        return regDataSet;
    }

//    public static RankDataSet loadRankDataSet(File trecFile, DataSetType dataSetType) throws Exception{
//        boolean legalArg = ((dataSetType == DataSetType.RANK_DENSE)
//                ||(dataSetType==DataSetType.RANK_SPARSE));
//        if (!legalArg){
//            throw new IllegalArgumentException("illegal data set type");
//        }
//        int numDataPoints = parseNumDataPoints(trecFile);
//        int numFeatures = parseNumFeaturess(trecFile);
//        RankDataSet rankDataSet = null;
//        if (dataSetType==DataSetType.RANK_DENSE){
//            rankDataSet = new DenseRankDataSet(numDataPoints,numFeatures);
//        }
//        if (dataSetType==DataSetType.RANK_SPARSE){
//            rankDataSet = new SparseRankDataSet(numDataPoints,numFeatures);
//        }
//        fillRankDataSet(rankDataSet,trecFile);
    //todo load settings
//        return rankDataSet;
//    }

    /**
     * parse comments in a TREC file
     * @param trecFile
     * @return
     * @throws Exception
     */
    public static List<String> parseComments(File trecFile) {
        List<String> comments = new ArrayList<>();
        File matrixFile = new File(trecFile, TREC_MATRIX_FILE_NAME);
        try (BufferedReader br = new BufferedReader(new FileReader(matrixFile));
        ) {
            String line = null;
            while ((line=br.readLine())!=null){
                //ignore the # sign itself
                int startPos = line.lastIndexOf("#")+1;
                int endPos = line.length();
                String comment = line.substring(startPos,endPos);
                System.out.println(comment);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return comments;
    }



    //==========PRIVATE==========

    private static int parseNumDataPoints(File trecFile) throws IOException {
        File configFile = new File(trecFile, TREC_CONFIG_FILE_NAME);
        int numDataPoints;
        try(
                BufferedReader br = new BufferedReader(new FileReader(configFile));
        ){
            Config config = new Config(configFile);
            numDataPoints = config.getInt(TREC_CONFIG_NUM_DATA_POINTS);
        }
        return numDataPoints;
    }

    private static int parseNumFeaturess(File trecFile) throws IOException {
        File configFile = new File(trecFile, TREC_CONFIG_FILE_NAME);
        int numFeatures;
        try(
                BufferedReader br = new BufferedReader(new FileReader(configFile));
        ){
            Config config = new Config(configFile);
            numFeatures = config.getInt(TREC_CONFIG_NUM_FEATURES);
        }
        return numFeatures;
    }

    private static void fillClfDataSet(ClfDataSet dataSet, File trecFile) throws IOException {
        File matrixFile = new File(trecFile, TREC_MATRIX_FILE_NAME);
        try (BufferedReader br = new BufferedReader(new FileReader(matrixFile));
        ) {
            String line = null;
            int dataIndex = 0;
            while ((line=br.readLine())!=null){
                String[] lineSplit = line.split(" ");
                int label = Integer.parseInt(lineSplit[0]);
                dataSet.setLabel(dataIndex,label);
                for (int i=1;i<lineSplit.length;i++){
                    String pair = lineSplit[i];
                    // ignore things after #
                    if (pair.startsWith("#")){
                        break;
                    }
                    String[] pairSplit = pair.split(":");
                    int featureIndex = Integer.parseInt(pairSplit[0]);
                    double featureValue = Double.parseDouble(pairSplit[1]);
                    dataSet.setFeatureValue(dataIndex, featureIndex,featureValue);
                }
                dataIndex += 1;
            }
        }
    }

    private static void fillRegDataSet(RegDataSet dataSet, File trecFile) throws IOException {
        File matrixFile = new File(trecFile, TREC_MATRIX_FILE_NAME);
        try (BufferedReader br = new BufferedReader(new FileReader(matrixFile));
        ) {
            String line = null;
            int dataIndex = 0;
            while ((line=br.readLine())!=null){
                String[] lineSplit = line.split(" ");
                double label = Double.parseDouble(lineSplit[0]);
                dataSet.setLabel(dataIndex,label);
                for (int i=1;i<lineSplit.length;i++){
                    String pair = lineSplit[i];
                    // ignore things after #
                    if (pair.startsWith("#")){
                        break;
                    }
                    String[] pairSplit = pair.split(":");
                    int featureIndex = Integer.parseInt(pairSplit[0]);
                    double featureValue = Double.parseDouble(pairSplit[1]);
                    dataSet.setFeatureValue(dataIndex, featureIndex,featureValue);
                }
                dataIndex += 1;
            }
        }
    }

    private static void fillRankDataSet(RankDataSet dataSet, File trecFile) throws IOException {
        File matrixFile = new File(trecFile, TREC_MATRIX_FILE_NAME);
        try (BufferedReader br = new BufferedReader(new FileReader(matrixFile));
        ) {
            String line = null;
            int dataIndex = 0;
            while ((line=br.readLine())!=null){
                String[] lineSplit = line.split(" ");
                int label = Integer.parseInt(lineSplit[0]);
                dataSet.setLabel(dataIndex,label);
                for (int i=1;i<lineSplit.length;i++){
                    String pair = lineSplit[i];
                    // ignore things after #
                    if (pair.startsWith("#")){
                        break;
                    }
                    String[] pairSplit = pair.split(":");
                    int featureIndex = Integer.parseInt(pairSplit[0]);
                    double featureValue = Double.parseDouble(pairSplit[1]);
                    dataSet.setFeatureValue(dataIndex, featureIndex,featureValue);
                }
                dataIndex += 1;
            }
        }
    }

    private static void writeConfigFile(DataSet dataSet, File trecFile) {
        File configFile = new File(trecFile, TREC_CONFIG_FILE_NAME);
        Config config = new Config();
        config.setInt(TREC_CONFIG_NUM_DATA_POINTS,dataSet.getNumDataPoints());
        config.setInt(TREC_CONFIG_NUM_FEATURES,dataSet.getNumFeatures());
        try {
            config.store(configFile);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void writeMatrixFile(ClfDataSet dataSet, File trecFile) {
        File matrixFile = new File(trecFile, TREC_MATRIX_FILE_NAME);
        int numDataPoints = dataSet.getNumDataPoints();
        int[] labels = dataSet.getLabels();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(matrixFile));
        ) {
            for (int i=0;i<numDataPoints;i++){
                int label = labels[i];
                bw.write(label+" ");
                FeatureRow featureRow = dataSet.getFeatureRow(i);
                Vector vector = featureRow.getVector();
                // only write non-zeros
                for (Vector.Element element: vector.nonZeroes()){
                    int featureIndex = element.index();
                    double featureValue = element.get();
                    bw.write(featureIndex+":"+featureValue+" ");
                }
                bw.write("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void writeMatrixFile(RegDataSet dataSet, File trecFile) {
        File matrixFile = new File(trecFile, TREC_MATRIX_FILE_NAME);
        int numDataPoints = dataSet.getNumDataPoints();
        double[] labels = dataSet.getLabels();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(matrixFile));
        ) {
            for (int i=0;i<numDataPoints;i++){
                double label = labels[i];
                bw.write(label+" ");
                FeatureRow featureRow = dataSet.getFeatureRow(i);
                Vector vector = featureRow.getVector();
                // only write non-zeros
                for (Vector.Element element: vector.nonZeroes()){
                    int featureIndex = element.index();
                    double featureValue = element.get();
                    bw.write(featureIndex+":"+featureValue+" ");
                }
                bw.write("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void writeDataSettings(DataSet dataSet, File trecFile) {
        File file = new File(trecFile,TREC_DATA_SETTINGS_FILE_NAME);
        List<DataSetting> dataSettings = new ArrayList<>();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            dataSettings.add(dataSet.getFeatureRow(i).getSetting());
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(dataSettings);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private static void writeFeatureSettings(DataSet dataSet, File trecFile) {
        File file = new File(trecFile,TREC_FEATURE_SETTINGS_FILE_NAME);
        List<FeatureSetting> featureSettings = new ArrayList<>();
        for (int i=0;i<dataSet.getNumFeatures();i++){
            featureSettings.add(dataSet.getFeatureColumn(i).getSetting());
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(featureSettings);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void loadDataSettings(DataSet dataSet, File trecFile) throws IOException, ClassNotFoundException {
        File file = new File(trecFile,TREC_DATA_SETTINGS_FILE_NAME);
        ArrayList<DataSetting> dataSettings;
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            dataSettings = (ArrayList)objectInputStream.readObject();

        }
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            dataSet.putDataSetting(i,dataSettings.get(i));
        }
    }

    private static void loadFeatureSettings(DataSet dataSet, File trecFile) throws IOException, ClassNotFoundException {
        File file = new File(trecFile,TREC_FEATURE_SETTINGS_FILE_NAME);
        ArrayList<FeatureSetting> featureSettings;
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            featureSettings = (ArrayList)objectInputStream.readObject();

        }
        for (int i=0;i<dataSet.getNumFeatures();i++){
            dataSet.putFeatureSetting(i,featureSettings.get(i));
        }
    }


}
