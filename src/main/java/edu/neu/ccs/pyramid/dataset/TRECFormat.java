package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by chengli on 8/19/14.
 */
public class TRECFormat {
    /**
     * internally, trecFile is a directory with several files in it
     */
    private static final String TREC_MATRIX_FILE_NAME = "feature_matrix.txt";
    private static final String TREC_CONFIG_FILE_NAME = "config.txt";
    private static final String TREC_CONFIG_NUM_DATA_POINTS = "numDataPoints";
    private static final String TREC_CONFIG_NUM_FEATURES = "numFeatures";
    private static final String TREC_CONFIG_NUM_CLASSES = "numClasses";
    private static final String TREC_CONFIG_MISSING_VALUE = "missingValue";
    private static final String TREC_FEATURE_LIST_FILE_NAME = "feature_list.ser";
    private static final String TREC_FEATURE_LIST_TEXT_FILE_NAME = "feature_list.txt";
    private static final String TREC_ID_TRANSLATOR_FILE_NAME = "id_translator.ser";
    private static final String TREC_ID_TRANSLATOR_TEXT_FILE_NAME = "id_translator.txt";
    private static final String TREC_LABEL_TRANSLATOR_FILE_NAME = "label_translator.ser";
    private static final String TREC_LABEL_TRANSLATOR_TEXT_FILE_NAME = "label_translator.txt";



    public static void save(DataSet dataSet, String trecFile){
        if (dataSet instanceof ClfDataSet){
            save((ClfDataSet)dataSet,trecFile);
        } else if (dataSet instanceof RegDataSet){
            save((RegDataSet) dataSet, trecFile);
        } else if (dataSet instanceof MultiLabelClfDataSet){
            save((MultiLabelClfDataSet) dataSet, trecFile);
        }
    }

    public static void save(ClfDataSet dataSet, String trecFile){
        save(dataSet,new File(trecFile));
    }

    public static void save(RegDataSet dataSet, String trecFile){
        save(dataSet,new File(trecFile));
    }

    public static void save(MultiLabelClfDataSet dataSet, String trecFile){
        save(dataSet, new File(trecFile));
    }

    public static void save(ClfDataSet dataSet, File trecFile){
        if (!trecFile.exists()){
            trecFile.mkdirs();
        }
        writeMatrixFile(dataSet, trecFile);
        writeConfigFile(dataSet, trecFile);

        writeFeatureList(dataSet, trecFile);
        writeIdTranslator(dataSet,trecFile);
        writeLabelTranslator(dataSet,trecFile);
    }

    public static void save(MultiLabelClfDataSet dataSet, File trecFile){
        if (!trecFile.exists()){
            trecFile.mkdirs();
        }
        writeMatrixFile(dataSet, trecFile);
        writeConfigFile(dataSet, trecFile);
        writeFeatureList(dataSet, trecFile);
        writeIdTranslator(dataSet, trecFile);
        writeLabelTranslator(dataSet, trecFile);
    }

    public static void save(RegDataSet dataSet, File trecFile) {
        if (!trecFile.exists()){
            trecFile.mkdirs();
        }
        writeMatrixFile(dataSet, trecFile);
        writeConfigFile(dataSet, trecFile);
        writeFeatureList(dataSet, trecFile);
        writeIdTranslator(dataSet, trecFile);

    }

    public static ClfDataSet loadClfDataSet(String trecFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return loadClfDataSet(new File(trecFile),dataSetType, loadSettings);
    }

    public static RegDataSet loadRegDataSet(String trecFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return loadRegDataSet(new File(trecFile),dataSetType, loadSettings);
    }

    public static MultiLabelClfDataSet loadMultiLabelClfDataSet(String trecFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return loadMultiLabelClfDataSet(new File(trecFile),dataSetType, loadSettings);
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
        int numClasses = parseNumClasses(trecFile);
        boolean missingValue = parseMissingValue(trecFile);
        ClfDataSet dataSet = null;
        if (dataSetType==DataSetType.CLF_DENSE){
            dataSet = new DenseClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        }
        if (dataSetType==DataSetType.CLF_SPARSE){
            dataSet = new SparseClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        }
        fillClfDataSet(dataSet,trecFile);
        if (loadSettings){
            loadFeatureList(dataSet,trecFile);
            loadIdTranslator(dataSet,trecFile);
            loadLabelTranslator(dataSet,trecFile);
        }

        return dataSet;
    }

    public static MultiLabelClfDataSet loadMultiLabelClfDataSet(File trecFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        boolean legalArg = ((dataSetType == DataSetType.ML_CLF_DENSE)
                ||(dataSetType==DataSetType.ML_CLF_SPARSE)||(dataSetType == DataSetType.ML_CLF_SEQ_SPARSE));
        if (!legalArg){
            throw new IllegalArgumentException("illegal data set type");
        }
        int numDataPoints = parseNumDataPoints(trecFile);
        int numFeatures = parseNumFeaturess(trecFile);
        int numClasses = parseNumClasses(trecFile);
        boolean missingValue = parseMissingValue(trecFile);
        MultiLabelClfDataSet dataSet = null;
        if (dataSetType==DataSetType.ML_CLF_DENSE){
            dataSet = new DenseMLClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        }
        if (dataSetType==DataSetType.ML_CLF_SPARSE){
            dataSet = new SparseMLClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        }
        if (dataSetType==DataSetType.ML_CLF_SEQ_SPARSE) {
            dataSet = new SequentialSparseMLClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        }
        fillMultiLabelClfDataSet(dataSet,trecFile);
        if (loadSettings){
            loadFeatureList(dataSet, trecFile);
            loadIdTranslator(dataSet, trecFile);
            loadLabelTranslator(dataSet, trecFile);
        }

        return dataSet;
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
        boolean missingValue = parseMissingValue(trecFile);
        RegDataSet dataSet = null;
        if (dataSetType==DataSetType.REG_DENSE){
            dataSet = new DenseRegDataSet(numDataPoints,numFeatures,missingValue);
        }
        if (dataSetType==DataSetType.REG_SPARSE){
            dataSet = new SparseRegDataSet(numDataPoints,numFeatures,missingValue);
        }
        fillRegDataSet(dataSet, trecFile);
        if (loadSettings){
            loadFeatureList(dataSet, trecFile);
            loadIdTranslator(dataSet, trecFile);
        }

        return dataSet;
    }


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

    private static int parseNumClasses(File trecFile) throws IOException {
        File configFile = new File(trecFile, TREC_CONFIG_FILE_NAME);
        int numClasses;
        try(
                BufferedReader br = new BufferedReader(new FileReader(configFile));
        ){
            Config config = new Config(configFile);
            numClasses = config.getInt(TREC_CONFIG_NUM_CLASSES);
        }
        return numClasses;
    }

    private static boolean parseMissingValue(File trecFile) throws IOException {
        File configFile = new File(trecFile, TREC_CONFIG_FILE_NAME);
        boolean missingValue;
        try(
                BufferedReader br = new BufferedReader(new FileReader(configFile));
        ){
            Config config = new Config(configFile);
            missingValue = config.getBoolean(TREC_CONFIG_MISSING_VALUE);
        }
        return missingValue;
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

    private static void fillMultiLabelClfDataSet(MultiLabelClfDataSet dataSet, File trecFile) throws IOException {
        File matrixFile = new File(trecFile, TREC_MATRIX_FILE_NAME);
        try (BufferedReader br = new BufferedReader(new FileReader(matrixFile));
        ) {
            String line = null;
            int dataIndex = 0;
            while ((line=br.readLine())!=null){
                String[] lineSplit = line.split(" ");
                String multiLabelString = null;
                try {
                    multiLabelString = lineSplit[0];
                } catch (Exception e) {
                    System.out.println("load data error happens");
                    System.out.println("line number: " + dataIndex);
                    System.out.println("line: " + line);
                }
                String[] multiLabelSplit = multiLabelString.split(Pattern.quote(","));
                for (String label: multiLabelSplit){
                    if (label.equals("")) {
                        continue;
                    }
                    dataSet.addLabel(dataIndex,Integer.parseInt(label));
                }
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





    private static void writeConfigFile(ClfDataSet dataSet, File trecFile) {
        File configFile = new File(trecFile, TREC_CONFIG_FILE_NAME);
        Config config = new Config();
        config.setInt(TREC_CONFIG_NUM_DATA_POINTS,dataSet.getNumDataPoints());
        config.setInt(TREC_CONFIG_NUM_FEATURES,dataSet.getNumFeatures());
        config.setInt(TREC_CONFIG_NUM_CLASSES,dataSet.getNumClasses());
        config.setBoolean(TREC_CONFIG_MISSING_VALUE,dataSet.hasMissingValue());
        try {
            config.store(configFile);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void writeConfigFile(MultiLabelClfDataSet dataSet, File trecFile) {
        File configFile = new File(trecFile, TREC_CONFIG_FILE_NAME);
        Config config = new Config();
        config.setInt(TREC_CONFIG_NUM_DATA_POINTS,dataSet.getNumDataPoints());
        config.setInt(TREC_CONFIG_NUM_FEATURES,dataSet.getNumFeatures());
        config.setInt(TREC_CONFIG_NUM_CLASSES,dataSet.getNumClasses());
        config.setBoolean(TREC_CONFIG_MISSING_VALUE,dataSet.hasMissingValue());
        try {
            config.store(configFile);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void writeConfigFile(RegDataSet dataSet, File trecFile) {
        File configFile = new File(trecFile, TREC_CONFIG_FILE_NAME);
        Config config = new Config();
        config.setInt(TREC_CONFIG_NUM_DATA_POINTS,dataSet.getNumDataPoints());
        config.setInt(TREC_CONFIG_NUM_FEATURES,dataSet.getNumFeatures());
        config.setBoolean(TREC_CONFIG_MISSING_VALUE,dataSet.hasMissingValue());
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
                bw.write("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void writeMatrixFile(MultiLabelClfDataSet dataSet, File trecFile) {
        File matrixFile = new File(trecFile, TREC_MATRIX_FILE_NAME);
        int numDataPoints = dataSet.getNumDataPoints();
        MultiLabel[] multiLabels = dataSet.getMultiLabels();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(matrixFile));
        ) {
            for (int i=0;i<numDataPoints;i++){
                MultiLabel multiLabel = multiLabels[i];
                List<Integer> labels = multiLabel.getMatchedLabels().stream().sorted().collect(Collectors.toList());
                for (int l=0;l<labels.size();l++){
                    bw.write(labels.get(l).toString());
                    if (l!=labels.size()-1){
                        bw.write(",");
                    }
                }
                bw.write(" ");
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
                bw.write("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }





    private static void writeFeatureList(DataSet dataSet, File trecFile) {
        File file = new File(trecFile, TREC_FEATURE_LIST_FILE_NAME);
        FeatureList featureList = dataSet.getFeatureList();
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(featureList);
        } catch (IOException e) {
            e.printStackTrace();
        }
        File txtFile = new File(trecFile, TREC_FEATURE_LIST_TEXT_FILE_NAME);
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(txtFile))
        ){
            for (Feature feature: featureList.getAll()){
                bufferedWriter.write(feature.toString());
                bufferedWriter.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    private static void loadFeatureList(DataSet dataSet, File trecFile) throws IOException, ClassNotFoundException {
        File file = new File(trecFile, TREC_FEATURE_LIST_FILE_NAME);
        if (file.exists()){
            FeatureList featureList;
            try(
                    FileInputStream fileInputStream = new FileInputStream(file);
                    BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                    ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
            ){
                featureList = (FeatureList)objectInputStream.readObject();

            }
            dataSet.setFeatureList(featureList);
        }

    }

    private static void writeIdTranslator(DataSet dataSet, File trecFile) {
        File file = new File(trecFile, TREC_ID_TRANSLATOR_FILE_NAME);
        IdTranslator idTranslator = dataSet.getIdTranslator();
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(idTranslator);
        } catch (IOException e) {
            e.printStackTrace();
        }
        File txtFile = new File(trecFile, TREC_ID_TRANSLATOR_TEXT_FILE_NAME);
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(txtFile))
        ){
            bufferedWriter.write(idTranslator.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    private static void loadIdTranslator(DataSet dataSet, File trecFile) throws IOException, ClassNotFoundException {
        File file = new File(trecFile, TREC_ID_TRANSLATOR_FILE_NAME);
        if (file.exists()){
            IdTranslator idTranslator;
            try(
                    FileInputStream fileInputStream = new FileInputStream(file);
                    BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                    ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
            ){
                idTranslator = (IdTranslator)objectInputStream.readObject();

            }
            dataSet.setIdTranslator(idTranslator);
        }

    }

    private static void writeLabelTranslator(ClfDataSet dataSet, File trecFile) {
        File file = new File(trecFile, TREC_LABEL_TRANSLATOR_FILE_NAME);
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(labelTranslator);
        } catch (IOException e) {
            e.printStackTrace();
        }

        File txtFile = new File(trecFile, TREC_LABEL_TRANSLATOR_TEXT_FILE_NAME);
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(txtFile))
        ){
            bufferedWriter.write(labelTranslator.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    private static void loadLabelTranslator(ClfDataSet dataSet, File trecFile) throws IOException, ClassNotFoundException {
        File file = new File(trecFile, TREC_LABEL_TRANSLATOR_FILE_NAME);
        if (file.exists()){
            LabelTranslator labelTranslator;
            try(
                    FileInputStream fileInputStream = new FileInputStream(file);
                    BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                    ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
            ){
                labelTranslator = (LabelTranslator)objectInputStream.readObject();

            }
            dataSet.setLabelTranslator(labelTranslator);
        }

    }

    private static void writeLabelTranslator(MultiLabelClfDataSet dataSet, File trecFile) {
        File file = new File(trecFile, TREC_LABEL_TRANSLATOR_FILE_NAME);
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(labelTranslator);
        } catch (IOException e) {
            e.printStackTrace();
        }
        File txtFile = new File(trecFile, TREC_LABEL_TRANSLATOR_TEXT_FILE_NAME);
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(txtFile))
        ){
            bufferedWriter.write(labelTranslator.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    private static void loadLabelTranslator(MultiLabelClfDataSet dataSet, File trecFile) throws IOException, ClassNotFoundException {
        File file = new File(trecFile, TREC_LABEL_TRANSLATOR_FILE_NAME);
        if (file.exists()){
            LabelTranslator labelTranslator;
            try(
                    FileInputStream fileInputStream = new FileInputStream(file);
                    BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                    ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
            ){
                labelTranslator = (LabelTranslator)objectInputStream.readObject();

            }
            dataSet.setLabelTranslator(labelTranslator);
        }

    }


}
