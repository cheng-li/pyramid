package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

public class IMLGBIsotonicScaling {
    private IsotonicRegression isotonicRegression;
    IMLGradientBoosting boosting;

    public IMLGBIsotonicScaling(IMLGradientBoosting boosting, MultiLabelClfDataSet multiLabelClfDataSet) {
        System.out.println("calibrating");
        this.boosting = boosting;
        double[] locations = new double[multiLabelClfDataSet.getNumDataPoints()];
        double[] binaryLabels = new double[multiLabelClfDataSet.getNumDataPoints()];

        SubsetAccPredictor predictor = new SubsetAccPredictor(boosting);
        IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                .forEach(i->{
                    MultiLabel pre = predictor.predict(multiLabelClfDataSet.getRow(i));
                    double score = boosting.predictAssignmentScore(multiLabelClfDataSet.getRow(i),pre);
                    locations[i] = score;
                    if (pre.equals(multiLabelClfDataSet.getMultiLabels()[i])){
                        binaryLabels[i] = 1;
                    } else {
                       binaryLabels[i] = 0;
                    }
                });
        isotonicRegression = new IsotonicRegression(locations, binaryLabels);
        System.out.println("calibration done");
    }

    public double calibratedProb(Vector vector, MultiLabel multiLabel){
        double score = boosting.predictAssignmentScore(vector, multiLabel);
        return isotonicRegression.predict(score);
    }
}
