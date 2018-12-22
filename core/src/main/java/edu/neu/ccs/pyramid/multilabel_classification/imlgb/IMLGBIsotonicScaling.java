package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;
import scala.Serializable;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class IMLGBIsotonicScaling implements Serializable{
    private static final long serialVersionUID = 1L;
    private IsotonicRegression isotonicRegression;
    private IMLGradientBoosting boosting;

    public IMLGBIsotonicScaling(IMLGradientBoosting boosting, MultiLabelClfDataSet multiLabelClfDataSet) {
//        System.out.println("calibrating with isotonic regression");
        this.boosting = boosting;
        List<MultiLabel> allAssignments = boosting.getAssignments();
        Stream<Pair<Double,Integer>> stream =  IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                .boxed().flatMap(i-> {
                    double[] probs = boosting.predictAllAssignmentProbsWithConstraint(multiLabelClfDataSet.getRow(i));
                    Stream<Pair<Double,Integer>> pairs = IntStream.range(0, probs.length).mapToObj(a -> {
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
        isotonicRegression = new IsotonicRegression(stream);
    }


    public IMLGradientBoosting getBoosting() {
        return boosting;
    }

    public IsotonicRegression getIsotonicRegression(){return isotonicRegression;}

    public double calibratedProb(Vector vector, MultiLabel multiLabel){
        double uncalibrated = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
        return isotonicRegression.predict(uncalibrated);
    }

    public double calibratedProb(double uncalibratedProb){
        return isotonicRegression.predict(uncalibratedProb);
    }


}
