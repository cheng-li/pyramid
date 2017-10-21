package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import org.apache.mahout.math.Vector;
import scala.Serializable;

import java.util.List;
import java.util.stream.IntStream;

public class IMLGBIsotonicScaling implements Serializable{
    private static final long serialVersionUID = 1L;
    private IsotonicRegression isotonicRegression;
    private IMLGradientBoosting boosting;

    public IMLGBIsotonicScaling(IMLGradientBoosting boosting, MultiLabelClfDataSet multiLabelClfDataSet) {
//        System.out.println("calibrating with isotonic regression");
        this.boosting = boosting;
        List<MultiLabel> allAssignments = boosting.getAssignments();
        double[] locations = new double[multiLabelClfDataSet.getNumDataPoints()*allAssignments.size()];
        double[] binaryLabels = new double[multiLabelClfDataSet.getNumDataPoints()*allAssignments.size()];

        SubsetAccPredictor predictor = new SubsetAccPredictor(boosting);
        IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                .forEach(i->{
                    double[] probs = boosting.predictAllAssignmentProbsWithConstraint(multiLabelClfDataSet.getRow(i));
                    for (int a=0;a<probs.length;a++){
                        locations[i*allAssignments.size()+a] = probs[a];
                        if (allAssignments.get(a).equals(multiLabelClfDataSet.getMultiLabels()[i])){
                            binaryLabels[i*allAssignments.size()+a] = 1;
                        } else {
                            binaryLabels[i*allAssignments.size()+a] = 0;
                        }
                    }
                });
        isotonicRegression = new IsotonicRegression(locations, binaryLabels);
//        System.out.println("calibration done");
    }

    public double calibratedProb(Vector vector, MultiLabel multiLabel){
        double uncalibrated = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
        return isotonicRegression.predict(uncalibrated);
    }

    public double calibratedProb(double uncalibratedProb){
        return isotonicRegression.predict(uncalibratedProb);
    }
}
