package edu.neu.ccs.pyramid.calibration;


import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.Vectors;

import java.util.List;
import java.util.stream.IntStream;

public class LabelProbUtil {

    public static LabelProbMatrix sampleData(LabelProbMatrix labelProbMatrix, List<Integer> indices){
        LabelProbMatrix sampled = new LabelProbMatrix();
        sampled.matrix = DataSetUtil.sampleData(labelProbMatrix.matrix,indices);
        sampled.labelTranslator = labelProbMatrix.labelTranslator;
        return sampled;
    }


    public static LabelProbMatrix calibrate(LabelProbMatrix labelProbMatrix, LabelCalibrator labelCalibrator,
                                            double threshold){
        LabelProbMatrix calibrated = new LabelProbMatrix(labelProbMatrix.getMatrix().getNumDataPoints(), labelProbMatrix.getMatrix().getNumFeatures(),
                labelProbMatrix.getLabelTranslator());
        IntStream.range(0, labelProbMatrix.getNumDataPoints()).parallel()
                .forEach(i->{
                    double[] rawProbs = Vectors.toArray(labelProbMatrix.getMatrix().getRow(i));
                    double[] calibratedProbs = labelCalibrator.calibratedClassProbs(rawProbs);
                    for (int j = 0; j < calibratedProbs.length; j++) {
                        if (calibratedProbs[j] >= threshold) {
                            labelProbMatrix.getMatrix().setFeatureValue(i,j,calibratedProbs[j]);
                        }
                    }
                });
        return calibrated;
    }


    public static LabelProbMatrix genLabelProbMatrix(MultiLabelClassifier.ClassProbEstimator classProbEstimator,
                                       MultiLabelClfDataSet dataSet,  double threshold){
        LabelProbMatrix labelProbMatrix = new LabelProbMatrix(dataSet.getNumDataPoints(), classProbEstimator.getNumClasses(),
                classProbEstimator.getLabelTranslator());
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(i->{
                    double[] classProbs = classProbEstimator.predictClassProbs(dataSet.getRow(i));
                    for (int j = 0; j < classProbs.length; j++) {
                        if (classProbs[j] >= threshold) {
                            labelProbMatrix.getMatrix().setFeatureValue(i,j,classProbs[j]);
                        }
                    }
                });
        return labelProbMatrix;
    }
}