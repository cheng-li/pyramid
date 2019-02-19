package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Pair;


import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class SetCalibrator implements Serializable {
    private static final long serialVersionUID = 1L;
    private IsotonicRegression isotonicRegression;

    public SetCalibrator(MultiLabelClassifier.AssignmentProbEstimator multiLabelClassifier, MultiLabelClfDataSet multiLabelClfDataSet,
                         List<MultiLabel> support) {
        Stream<Pair<Double,Integer>> stream =  IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                .boxed().flatMap(i-> {
                    double[] probs = multiLabelClassifier.predictAssignmentProbs(multiLabelClfDataSet.getRow(i),support);
                    Stream<Pair<Double,Integer>> pairs = IntStream.range(0, probs.length).mapToObj(a -> {
                        Pair<Double, Integer> pair = new Pair<>();
                        pair.setFirst(probs[a]);
                        pair.setSecond(0);
                        if (support.get(a).equals(multiLabelClfDataSet.getMultiLabels()[i])) {
                            pair.setSecond(1);
                        }
                        return pair;
                    });
                    return pairs;
                });
        isotonicRegression = new IsotonicRegression(stream);
    }

    public double calibrate(double uncalibrated){
        return isotonicRegression.predict(uncalibrated);
    }
}
