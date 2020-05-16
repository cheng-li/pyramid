package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.Density;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.Vector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class LabelProbMatrix {
    private LabelTranslator labelTranslator;
    private DataSet matrix;

    public LabelProbMatrix(int numRows, int numColumns, LabelTranslator labelTranslator) {
        this.matrix = DataSetBuilder.getBuilder()
                .numDataPoints(numRows)
                .numFeatures(numColumns)
                .density(Density.SPARSE_RANDOM)
                .build();
        this.labelTranslator = labelTranslator;
    }

    public LabelProbMatrix(File labelProbMatrixFile, File labelTranslatorFile){
        this.labelTranslator = LabelTranslator.loadFromTxtFile(labelTranslatorFile);
        try {
            this.matrix = loadProbMatrix(labelProbMatrixFile,labelTranslator.getNumClasses());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public LabelProbMatrix(File labelProbMatrixFile, LabelTranslator labelTranslator){
        this.labelTranslator = labelTranslator;
        try {
            this.matrix = loadProbMatrix(labelProbMatrixFile,labelTranslator.getNumClasses());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public LabelProbMatrix(LabelTranslator labelTranslator, DataSet matrix){
        this.labelTranslator = labelTranslator;
        try {
            this.matrix = matrix;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }




    public Optional<Vector> getLabelProbForAllInstances(String label){
        if (labelTranslator.getAllExtLabels().contains(label)){
            int labelId = labelTranslator.toIntLabel(label);
            return Optional.of(matrix.getColumn(labelId));
        } else return Optional.empty();
    }

    private static DataSet loadProbMatrix(File labelProbMatrixFile, int numLabels) throws Exception{
        List<String> lines = FileUtils.readLines(labelProbMatrixFile);
        DataSet matrix = DataSetBuilder.getBuilder()
                .numDataPoints(lines.size())
                .numFeatures(numLabels)
                .density(Density.SPARSE_RANDOM)
                .build();
        for (int i=0;i<lines.size();i++){
            String line = lines.get(i);
            if (!line.equals(" ")){
                String[] split = line.split(" ");
                for (String pair: split){
                    if (!pair.isEmpty()){
                        int labelIndex = Integer.parseInt(pair.split(":")[0]);
                        double prob = Double.parseDouble(pair.split(":")[1]);
                        matrix.setFeatureValue(i,labelIndex,prob);
                    }

                }
            }
        }
        return matrix;
    }


    public int getNumDataPoints() {
        return matrix.getNumDataPoints();
    }
    public LabelTranslator getLabelTranslator(){
        return labelTranslator;
    }
    public DataSet getMatrix(){
        return matrix;
    }

    public int getNumLabels(){
        return labelTranslator.getNumClasses();
    }

    public void writeToFile(File file){
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file));
        ) {
            for (int i=0;i<matrix.getNumDataPoints();i++){
                Vector vector = matrix.getRow(i);
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



}
