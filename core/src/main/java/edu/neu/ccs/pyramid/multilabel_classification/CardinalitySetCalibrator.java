package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.ArgMin;
import edu.neu.ccs.pyramid.util.Pair;


import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class CardinalitySetCalibrator implements Serializable {
    private static final long serialVersionUID = 1L;
    Map<Integer,IsotonicRegression> calibrations;

    public CardinalitySetCalibrator(MultiLabelClassifier.AssignmentProbEstimator multiLabelClassifier,
                                    MultiLabelClfDataSet multiLabelClfDataSet,
                                    List<MultiLabel> support) {
        this.calibrations = new HashMap<>();
        Set<Integer> cardinalities = new HashSet<>();
        for (MultiLabel multiLabel : support) {
            cardinalities.add(multiLabel.getNumMatchedLabels());
        }

        for (int cardinality : cardinalities) {
            Stream<Pair<Double, Double>> stream = IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                    .boxed().flatMap(i -> {
                        double[] probs = multiLabelClassifier.predictAssignmentProbs(multiLabelClfDataSet.getRow(i), support);
                        Stream<Pair<Double, Double>> pairs = IntStream.range(0, probs.length)
                                .filter(a -> support.get(a).getNumMatchedLabels() == cardinality)
                                .mapToObj(a -> {
                                    Pair<Double, Double> pair = new Pair<>();
                                    pair.setFirst(probs[a]);
                                    pair.setSecond(0.0);
                                    if (support.get(a).equals(multiLabelClfDataSet.getMultiLabels()[i])) {
                                        pair.setSecond(1.0);
                                    }
                                    return pair;
                                });
                        return pairs;
                    });
            calibrations.put(cardinality, new IsotonicRegression(stream));
        }
    }

    public double calibrate(double uncalibrated, int cardinality){
        //deal with unseen cardinality
        List<Integer> cards = new ArrayList<>(calibrations.keySet());
        Collections.sort(cards);
        double[] diff = new double[cards.size()];
        for (int i=0;i<cards.size();i++){
            diff[i] = Math.abs(cardinality-cards.get(i));
        }
        int closest = cards.get(ArgMin.argMin(diff));
        return calibrations.get(closest).predict(uncalibrated);
    }
}
