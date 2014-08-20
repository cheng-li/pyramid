package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.Vector;

import java.io.*;

/**
 * Created by chengli on 8/20/14.
 */
public class StandardFormat {
    public static ClfDataSet loadClfDataSet(String featureFile,
                                            String labelFile,
                                            String delimiter,
                                            DataSetType dataSetType) throws IOException {
        return loadClfDataSet(new File(featureFile),
                new File(labelFile),delimiter,dataSetType);
    }

    public static RegDataSet loadRegDataSet(String featureFile,
                                            String labelFile,
                                            String delimiter,
                                            DataSetType dataSetType) throws IOException {
        return loadRegDataSet(new File(featureFile),
                new File(labelFile), delimiter, dataSetType);
    }


    public static ClfDataSet loadClfDataSet(File featureFile,
                                            File labelFile,
                                            String delimiter,
                                            DataSetType dataSetType) throws IOException {
        int[] stats = parseStandard(featureFile, labelFile, delimiter);
        int numDataPoints = stats[0];
        int numFeatures = stats[1];
        ClfDataSet dataSet = null;
        boolean legalArg = ((dataSetType == DataSetType.CLF_DENSE)
                ||(dataSetType==DataSetType.CLF_SPARSE));
        if (!legalArg){
            throw new IllegalArgumentException("illegal data set type");
        }
        if (dataSetType==DataSetType.CLF_DENSE){
            dataSet = new DenseClfDataSet(numDataPoints,numFeatures);
        }
        if (dataSetType==DataSetType.CLF_SPARSE){
            dataSet = new SparseClfDataSet(numDataPoints,numFeatures);
        }
        fill(dataSet,featureFile,labelFile,delimiter);
        return dataSet;
    }

    public static RegDataSet loadRegDataSet(File featureFile,
                                            File labelFile,
                                            String delimiter,
                                            DataSetType dataSetType) throws IOException {
        int[] stats = parseStandard(featureFile, labelFile, delimiter);
        int numDataPoints = stats[0];
        int numFeatures = stats[1];
        RegDataSet dataSet = null;
        boolean legalArg = ((dataSetType == DataSetType.REG_DENSE)
                ||(dataSetType==DataSetType.REG_SPARSE));
        if (!legalArg){
            throw new IllegalArgumentException("illegal data set type");
        }
        if (dataSetType==DataSetType.REG_DENSE){
            dataSet = new DenseRegDataSet(numDataPoints,numFeatures);
        }
        if (dataSetType==DataSetType.REG_SPARSE){
            dataSet = new SparseRegDataSet(numDataPoints,numFeatures);
        }
        fill(dataSet,featureFile,labelFile,delimiter);
        return dataSet;
    }

    //todo save settings?

    public static void save(ClfDataSet dataSet,
                            File featureFile,
                            File labelFile,
                            String delimiter) throws Exception{
        int numDataPoints = dataSet.getNumDataPoints();
        int numFeatures = dataSet.getNumFeatures();
        try(
                BufferedWriter bw1 = new BufferedWriter(new FileWriter(featureFile));
        ){
            for (int i=0;i<numDataPoints;i++){
                FeatureRow featureRow = dataSet.getFeatureRow(i);
                Vector vector = featureRow.getVector();
                for (int j=0;j<numFeatures;j++){
                    bw1.write(""+vector.get(j));
                    if (j!=numFeatures-1){
                        bw1.write(delimiter);
                    } else {
                        bw1.write("\n");
                    }
                }
            }
        }

        try(
                BufferedWriter bw2 = new BufferedWriter(new FileWriter(labelFile));
        ){
            int[] labels = dataSet.getLabels();
            for (int i=0;i<numDataPoints;i++){
                bw2.write(""+labels[i]);
                bw2.write("\n");
            }
        }
    }

    public static void save(RegDataSet dataSet,
                            File featureFile,
                            File labelFile,
                            String delimiter) throws Exception{
        int numDataPoints = dataSet.getNumDataPoints();
        int numFeatures = dataSet.getNumFeatures();
        try(
                BufferedWriter bw1 = new BufferedWriter(new FileWriter(featureFile));
        ){
            for (int i=0;i<numDataPoints;i++){
                FeatureRow featureRow = dataSet.getFeatureRow(i);
                Vector vector = featureRow.getVector();
                for (int j=0;j<numFeatures;j++){
                    bw1.write(""+vector.get(j));
                    if (j!=numFeatures-1){
                        bw1.write(delimiter);
                    } else {
                        bw1.write("\n");
                    }
                }
            }
        }

        try(
                BufferedWriter bw2 = new BufferedWriter(new FileWriter(labelFile));
        ){
            double[] labels = dataSet.getLabels();
            for (int i=0;i<numDataPoints;i++){
                bw2.write(""+labels[i]);
                bw2.write("\n");
            }
        }
    }

    private static int[] parseStandard(File featureFile, File labelFile, String delimiter) throws IOException {
        int numDataPoints=0;
        try(
                BufferedReader br1 = new BufferedReader((new FileReader(labelFile)));
        ){
            //TODO: make it more robust to empty end lines
            while(br1.readLine()!=null){
                numDataPoints += 1;
            }
        }

        int numFeatures=0;
        try(
                BufferedReader br2 = new BufferedReader(new FileReader(featureFile));
        ){
            String line = br2.readLine();
            numFeatures = line.split(delimiter).length;
        }

        int[] stats = {numDataPoints,numFeatures};
        return stats;
    }

    /**
     * fill up a reg data set
     * @param regDataSet
     * @param featureFile
     * @param labelFile
     * @param delimiter
     * @throws Exception
     */
    private static void fill(RegDataSet regDataSet,
                             File featureFile,
                             File labelFile,
                             String delimiter) throws IOException {
        int numFeatures = regDataSet.getNumFeatures();

        try (
                BufferedReader br1 = new BufferedReader(new FileReader(featureFile));
        ){
            int i = 0;
            String line1 = null;
            while((line1=br1.readLine())!=null){
                String[] line1Split = line1.split(delimiter);
                for (int j=0;j<numFeatures;j++){
                    double featureValue = Double.parseDouble(line1Split[j]);
                    regDataSet.setFeatureValue(i, j, featureValue);
                }
                i += 1;
            }
        }

        try (BufferedReader br2 = new BufferedReader((new FileReader(labelFile)));
        ){
            String line2 = null;
            int k = 0;
            while((line2=br2.readLine())!=null){
                String[] line2Split = line2.split(delimiter);
                double label = Double.parseDouble(line2Split[0]);
                regDataSet.setLabel(k, label);
                k += 1;
            }
        }
    }

    /**
     * fill up a clf data set
     * @param clfDataSet
     * @param featureFile
     * @param labelFile
     * @param delimiter
     * @throws Exception
     */
    private static void fill(ClfDataSet clfDataSet,
                             File featureFile,
                             File labelFile,
                             String delimiter) throws IOException {
        int numFeatures = clfDataSet.getNumFeatures();

        try (
                BufferedReader br1 = new BufferedReader(new FileReader(featureFile));
        ){
            int i = 0;
            String line1 = null;
            while((line1=br1.readLine())!=null){
                String[] line1Split = line1.split(delimiter);
                for (int j=0;j<numFeatures;j++){
                    double featureValue = Double.parseDouble(line1Split[j]);
                    clfDataSet.setFeatureValue(i,j,featureValue);
                }
                i += 1;
            }
        }

        try (BufferedReader br2 = new BufferedReader((new FileReader(labelFile)));
        ){
            String line2 = null;
            int k = 0;
            while((line2=br2.readLine())!=null){
                String[] line2Split = line2.split(delimiter);
                int label = Integer.parseInt(line2Split[0]);
                clfDataSet.setLabel(k,label);
                k += 1;
            }
        }
    }
}
