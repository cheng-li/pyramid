package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 4/3/17.
 */
public class MarginalPredictor implements PluginPredictor<CBM> {
    CBM cbm;
    double piThreshold =0.001;

    public MarginalPredictor(CBM cbm) {
        this.cbm = cbm;
    }

    public void setPiThreshold(double piThreshold) {
        this.piThreshold = piThreshold;
    }

    @Override
    public CBM getModel() {
        return cbm;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        BMDistribution bmDistribution = new BMDistribution(cbm, vector, piThreshold);
        double[] probs = bmDistribution.marginals();
        MultiLabel prediction = new MultiLabel();
        for (int l=0;l<cbm.getNumClasses();l++){
            if (probs[l]>=0.5){
                prediction.addLabel(l);
            }
        }
        return prediction;
    }
}
