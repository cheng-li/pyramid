package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.AbstractPlugIn;
import org.apache.mahout.math.Vector;

/**
 * Optimal for subset accuracy
 * Created by chengli on 4/6/16.
 */
public class SubsetAccPredictor extends AbstractPlugIn {
    private static final long serialVersionUID = 1L;
    private IMLGradientBoosting imlGradientBoosting;

    public SubsetAccPredictor(IMLGradientBoosting imlGradientBoosting) {
        super(imlGradientBoosting);
        this.imlGradientBoosting = imlGradientBoosting;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        if (imlGradientBoosting.getAssignments()==null){
            throw new RuntimeException("CRF is used but legal assignments is not specified!");
        }
        double maxScore = Double.NEGATIVE_INFINITY;
        MultiLabel prediction = null;
        double[] classScores = imlGradientBoosting.predictClassScores(vector);
        for (MultiLabel assignment: imlGradientBoosting.getAssignments()){
            double score = imlGradientBoosting.calAssignmentScore(assignment,classScores);
            if (score > maxScore){
                maxScore = score;
                prediction = assignment;
            }
        }
        return prediction;
    }

    public double predictAssignmentProb(Vector vector, MultiLabel assignment){
        if (assignment.outOfBound(imlGradientBoosting.getNumClasses())){
            return 0;
        }
        return imlGradientBoosting.predictAssignmentProbWithConstraint(vector,assignment);
    }
}
