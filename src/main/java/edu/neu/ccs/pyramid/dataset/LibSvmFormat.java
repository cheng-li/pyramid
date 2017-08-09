package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by chengli on 10/28/14.
 */
public class LibSvmFormat {

    public static void save(ClfDataSet dataSet, String libSvmFile){
        File matrixFile = new File(libSvmFile);
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
                    Pair<Integer,Double> pair = new Pair<>(element.index()+1,element.get());
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

    public static void save(RegDataSet dataSet, String libSvmFile){
        File matrixFile = new File(libSvmFile);
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
                    Pair<Integer,Double> pair = new Pair<>(element.index()+1,element.get());
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

    public static void save(MultiLabelClfDataSet dataSet, String libSvmFile){
        File matrixFile = new File(libSvmFile);
        int numDataPoints = dataSet.getNumDataPoints();
        MultiLabel[] multiLabels = dataSet.getMultiLabels();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(matrixFile));) {
            for (int i=0;i<numDataPoints;i++){
                MultiLabel multiLabel = multiLabels[i];
                List<Integer> labels = multiLabel.getMatchedLabels().stream().sorted().collect(Collectors.toList());
                for (int l=0;l<labels.size();l++){
                    bw.write(labels.get(l).toString());
                    if (l!=labels.size()-1){
                        bw.write(",");
                    } else {
                        bw.write(" ");
                    }
                }
                Vector vector = dataSet.getRow(i);
                // only write non-zeros
                List<Pair<Integer,Double>> pairs = new ArrayList<>();
                for (Vector.Element element:vector.nonZeroes()){
                    Pair<Integer,Double> pair = new Pair<>(element.index()+1,element.get());
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

    public static ClfDataSet loadClfDataSet(String libSvmFile,
                                            int numFeatures, int numClasses, boolean dense) throws IOException, ClassNotFoundException {
        LabelTranslator labelTranslator = loadLabelTranslator(libSvmFile);
        System.out.println(labelTranslator);

        if (labelTranslator.getNumClasses()!=numClasses){
            throw new RuntimeException("labelTranslator.getNumClasses()!=numClasse");
        }

        int numDataPoints = getNumDataPoints(libSvmFile);

        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(numFeatures)
                .numClasses(numClasses)
                .dense(dense)
                .build();
        try (BufferedReader br = new BufferedReader(new FileReader(libSvmFile));
        ) {
            String line = null;
            int dataIndex = 0;
            while ((line=br.readLine())!=null){
                // sometimes, they use more than one spaces
                String[] lineSplit = line.split("\\s+");
                int extIntLabel = (int)Double.parseDouble(lineSplit[0]);
                String extLabel = ""+extIntLabel;
                int label = labelTranslator.toIntLabel(extLabel);
                dataSet.setLabel(dataIndex,label);
                for (int i=1;i<lineSplit.length;i++){
                    String pair = lineSplit[i];
                    // ignore things after #
                    if (pair.startsWith("#")){
                        break;
                    }
                    String[] pairSplit = pair.split(":");
                    int featureIndex = Integer.parseInt(pairSplit[0])-1;
                    double featureValue = Double.parseDouble(pairSplit[1]);
                    dataSet.setFeatureValue(dataIndex, featureIndex,featureValue);
                }
                dataIndex += 1;
            }
        }
        dataSet.setLabelTranslator(labelTranslator);
        return dataSet;

    }

    public static RegDataSet loadRegDataSet(String libSvmFile,
                                            int numFeatures, boolean dense) throws IOException, ClassNotFoundException {
        int numDataPoints = getNumDataPoints(libSvmFile);

        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(numFeatures)
                .dense(dense)
                .build();
        try (BufferedReader br = new BufferedReader(new FileReader(libSvmFile));
        ) {
            String line = null;
            int dataIndex = 0;
            while ((line=br.readLine())!=null){
                // sometimes, they use more than one spaces
                String[] lineSplit = line.split("\\s+");
                double label = Double.parseDouble(lineSplit[0]);

                dataSet.setLabel(dataIndex,label);
                for (int i=1;i<lineSplit.length;i++){
                    String pair = lineSplit[i];
                    // ignore things after #
                    if (pair.startsWith("#")){
                        break;
                    }
                    String[] pairSplit = pair.split(":");
                    int featureIndex = Integer.parseInt(pairSplit[0])-1;
                    double featureValue = Double.parseDouble(pairSplit[1]);
                    dataSet.setFeatureValue(dataIndex, featureIndex,featureValue);
                }
                dataIndex += 1;
            }
        }
        return dataSet;
    }

    public static LabelTranslator loadLabelTranslator(String libSvmFile) throws IOException{
        Set<Integer> oldLabels = new HashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(libSvmFile));
        ) {
            String line = null;
            while ((line=br.readLine())!=null){
                String[] lineSplit = line.split("\\s+");
                int label = (int)Double.parseDouble(lineSplit[0]);
                oldLabels.add(label);
            }
        }
        List<String> labelStrings = oldLabels.stream().sorted().map(label -> ""+label).collect(Collectors.toList());
        LabelTranslator labelTranslator = new LabelTranslator(labelStrings);
        return labelTranslator;
    }

    public static int getNumDataPoints(String libSvmFile) throws IOException{
        int num = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(libSvmFile));
        ) {
            String line = null;
            while ((line=br.readLine())!=null){
                num += 1;
            }
            br.close();
        }
        return num;
    }

    public static RegDataSet loadRegDataSet(String libSvmFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }

    public static MultiLabelClfDataSet loadMultiLabelClfDataSet(String libSvmFile,
                                                                boolean dense, int numFeatures, int numClasses) throws IOException, ClassNotFoundException {
        int numDatapoints = getNumDataPoints(libSvmFile);
//        int numClasses = getNumClasses(libSvmFile);
//        int numFeatures = getnumFeatures(libSvmFile);

        System.out.println("numDatapoints: " + numDatapoints);
        System.out.println("numClasses: " + numClasses);
        System.out.println("numFeatures: " + numFeatures);

        MultiLabelClfDataSet dataSet = new MLClfDataSetBuilder().numClasses(numClasses)
                .numFeatures(numFeatures).numDataPoints(numDatapoints).
                        density(Density.SPARSE_RANDOM).build();

        try (BufferedReader br = new BufferedReader(new FileReader(libSvmFile));
        ) {
            String line = null;
            int lineCount = 0;
            while ((line=br.readLine()) != null) {
                String[] lineInfo = line.split(" ");
                // adding labels
                String labels = lineInfo[0];
                for (String label : labels.split(",")) {
                    int l = Integer.parseInt(label);
                    dataSet.addLabel(lineCount, l-1);
//                    dataSet.addLabel(lineCount, l);
                }
                // adding feature
                for (int i=1; i<lineInfo.length; i++) {
                    String[] featureValue = lineInfo[i].split(":");
                    int feature = Integer.parseInt(featureValue[0]);
                    double value = Double.parseDouble(featureValue[1]);
                    dataSet.setFeatureValue(lineCount, feature-1, value);
                }
                lineCount++;
            }
            br.close();
        }


        return dataSet;
    }

    private static int getnumFeatures(String libSvmFile) throws IOException {
        Set<String> featureSet = new HashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(libSvmFile));
        ) {
            String line = null;
            while ((line=br.readLine()) != null) {
                String[] features = line.split(" ");
                for (int i=1; i<features.length; i++) {
                    String featureIndex = features[i].split(":")[0];
                    if (!featureSet.contains(featureIndex)) {
                        featureSet.add(featureIndex);
                    }
                }
            }
            br.close();
        }
        return featureSet.size();
    }

    private static int getNumClasses(String libSvmFile) throws IOException {
        Set<String> labelSet = new HashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(libSvmFile));
        ) {
            String line = null;
            while ((line=br.readLine()) != null) {
                String labelStr = line.split(" ")[0];
                String[] labels = labelStr.split(",");
                for (String l : labels) {
                    if (!labelSet.contains(l)) {
                        labelSet.add(l);
                    }
                }
            }
            br.close();
        }

        return labelSet.size();
    }
}
