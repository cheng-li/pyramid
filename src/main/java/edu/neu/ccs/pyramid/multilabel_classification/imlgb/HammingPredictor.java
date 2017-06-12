package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import org.apache.mahout.math.Vector;

/**
 * optimal for Hamming Loss
 * Created by chengli on 4/6/16.
 */
public class HammingPredictor implements PluginPredictor<IMLGradientBoosting>{


    private static final long serialVersionUID = 1L;
    private IMLGradientBoosting imlGradientBoosting;


    public HammingPredictor(IMLGradientBoosting imlGradientBoosting) {
        this.imlGradientBoosting = imlGradientBoosting;
    }

    @Override
    public IMLGradientBoosting getModel() {
        return imlGradientBoosting;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        MultiLabel prediction = new MultiLabel();
        double[] scores = imlGradientBoosting.predictClassScoresCachedInput(vector);
        for (int k=0;k<getNumClasses();k++){
            if (scores[k] > 0){
                prediction.addLabel(k);
            }
        }
        return prediction;
    }

    public double predictAssignmentProb(Vector vector, MultiLabel assignment){
        if (assignment.outOfBound(imlGradientBoosting.getNumClasses())){
            return 0;
        }
        return imlGradientBoosting.predictAssignmentProbWithoutConstraint(vector,assignment);
    }

}
