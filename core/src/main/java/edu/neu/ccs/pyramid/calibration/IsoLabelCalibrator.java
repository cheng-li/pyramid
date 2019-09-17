package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class IsoLabelCalibrator implements LabelCalibrator {
    private static final long serialVersionUID = 1L;
    List<IsotonicRegression> isotonicRegressionList;


//    public IsoLabelCalibrator(MultiLabelClassifier.ClassProbEstimator multiLabelClassifier, MultiLabelClfDataSet multiLabelClfDataSet) {
////        this.isotonicRegressionList = new ArrayList<>();
////        for (int l= 0; l < multiLabelClassifier.getNumClasses(); l++) {
////            final int calssIndex = l;
////            Stream<Pair<Double,Double>> stream = IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
////                    .mapToObj(i->{
////                        double prob = multiLabelClassifier.predictClassProb(multiLabelClfDataSet.getRow(i), calssIndex);
////                        Pair<Double,Double> pair = new Pair<>();
////                        pair.setFirst(prob);
////                        pair.setSecond(0.0);
////                        if (multiLabelClfDataSet.getMultiLabels()[i].matchClass(calssIndex)){
////                            pair.setSecond(1.0);
////                        }
////                        return pair;
////                    });
////
////            IsotonicRegression isotonicRegression = new IsotonicRegression(stream);
////            isotonicRegressionList.add(isotonicRegression);
////        }
////    }

    public IsoLabelCalibrator(MultiLabelClassifier.ClassProbEstimator multiLabelClassifier, MultiLabelClfDataSet multiLabelClfDataSet) {
        this(multiLabelClassifier,multiLabelClfDataSet,false);
    }

    public IsoLabelCalibrator(MultiLabelClassifier.ClassProbEstimator multiLabelClassifier, MultiLabelClfDataSet multiLabelClfDataSet, boolean interpolate) {
        this.isotonicRegressionList = new ArrayList<>();
        for (int l= 0; l < multiLabelClassifier.getNumClasses(); l++) {
            final int calssIndex = l;
            Stream<Pair<Double,Double>> stream = IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                    .mapToObj(i->{
                        double prob = multiLabelClassifier.predictClassProb(multiLabelClfDataSet.getRow(i), calssIndex);
                        Pair<Double,Double> pair = new Pair<>();
                        pair.setFirst(prob);
                        pair.setSecond(0.0);
                        if (multiLabelClfDataSet.getMultiLabels()[i].matchClass(calssIndex)){
                            pair.setSecond(1.0);
                        }
                        return pair;
                    });

            IsotonicRegression isotonicRegression = new IsotonicRegression(stream,interpolate);
            isotonicRegressionList.add(isotonicRegression);
        }
    }

    public IsoLabelCalibrator(Vector[] probabilities, MultiLabelClfDataSet multiLabelClfDataSet, boolean interpolate) {
        this.isotonicRegressionList = new ArrayList<>();
        int numClasses = probabilities[0].size();
        for (int l= 0; l < numClasses; l++) {
            final int calssIndex = l;
            Stream<Pair<Double,Double>> stream = IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                    .mapToObj(i->{
                        double prob = probabilities[i].get(calssIndex);
                        Pair<Double,Double> pair = new Pair<>();
                        pair.setFirst(prob);
                        pair.setSecond(0.0);
                        if (multiLabelClfDataSet.getMultiLabels()[i].matchClass(calssIndex)){
                            pair.setSecond(1.0);
                        }
                        return pair;
                    });

            IsotonicRegression isotonicRegression = new IsotonicRegression(stream, interpolate);
            isotonicRegressionList.add(isotonicRegression);
        }
    }


    public IsoLabelCalibrator(List<Vector> probabilities, MultiLabelClfDataSet multiLabelClfDataSet, boolean interpolate) {
        this.isotonicRegressionList = new ArrayList<>();
        int numClasses = probabilities.get(0).size();
        for (int l= 0; l < numClasses; l++) {
            final int calssIndex = l;
            Stream<Pair<Double,Double>> stream = IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                    .mapToObj(i->{
                        double prob = probabilities.get(i).get(calssIndex);
                        Pair<Double,Double> pair = new Pair<>();
                        pair.setFirst(prob);
                        pair.setSecond(0.0);
                        if (multiLabelClfDataSet.getMultiLabels()[i].matchClass(calssIndex)){
                            pair.setSecond(1.0);
                        }
                        return pair;
                    });

            IsotonicRegression isotonicRegression = new IsotonicRegression(stream, interpolate);
            isotonicRegressionList.add(isotonicRegression);
        }
    }


    public double calibratedClassProb(double prob, int labelIndex){
        return isotonicRegressionList.get(labelIndex).predict(prob);
    }

    public double[] calibratedClassProbs(double[]probs){
        return IntStream.range(0, probs.length).mapToDouble(j->calibratedClassProb(probs[j], j)).toArray();

    }

    public List<IsotonicRegression> getIsotonicRegressionList() {
        return isotonicRegressionList;
    }
}
