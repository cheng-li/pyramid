package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.util.Pair;
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
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(matrixFile));
        ) {
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

    public static ClfDataSet loadClfDataSet(String libSvmFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }

    public static RegDataSet loadRegDataSet(String libSvmFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }

    public static MultiLabelClfDataSet loaMultiLabelClfDataSet(String libSvmFile, DataSetType dataSetType,
                                                               boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }
}
