package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class CardinalityCalibrator implements Serializable{
    private static final long serialVersionUID = 1L;
    Map<Integer,IsotonicRegression> calibrations;
    private IMLGradientBoosting boosting;

    public CardinalityCalibrator(IMLGradientBoosting boosting, MultiLabelClfDataSet multiLabelClfDataSet) {
        this.calibrations = new HashMap<>();
        List<MultiLabel> support = boosting.getAssignments();
        Set<Integer> cardinalities = new HashSet<>();
        for (MultiLabel multiLabel: support){
            cardinalities.add(multiLabel.getNumMatchedLabels());
        }

        for (int cardinality: cardinalities){
            List<MultiLabel> allAssignments = boosting.getAssignments();
            Stream<Pair<Double,Integer>> stream =  IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                    .boxed().flatMap(i-> {
                        double[] probs = boosting.predictAllAssignmentProbsWithConstraint(multiLabelClfDataSet.getRow(i));
                        Stream<Pair<Double,Integer>> pairs = IntStream.range(0, probs.length)
                                .filter(a->allAssignments.get(a).getNumMatchedLabels()==cardinality)
                                .mapToObj(a -> {
                            Pair<Double, Integer> pair = new Pair<>();
                            pair.setFirst(probs[a]);
                            pair.setSecond(0);
                            if (allAssignments.get(a).equals(multiLabelClfDataSet.getMultiLabels()[i])) {
                                pair.setSecond(1);
                            }
                            return pair;
                        });
                        return pairs;
                    });
            calibrations.put(cardinality, new IsotonicRegression(stream));
        }



//        System.out.println("calibrating with isotonic regression");
        this.boosting = boosting;

    }



    public double calibrate(double uncalibrated, int cardinality){
        return calibrations.get(cardinality).predict(uncalibrated);
    }


    public double calibratedProb(Vector vector, MultiLabel multiLabel){
        double uncalibrated = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
        int cardinality = multiLabel.getNumMatchedLabels();
        return calibrations.get(cardinality).predict(uncalibrated);
    }


}
