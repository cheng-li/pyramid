package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 3/28/17.
 */
public class AccPredictor implements PluginPredictor<CBM> {
    CBM cbm;
    private double componentContributionThreshold = 0;

    public AccPredictor(CBM cbm) {
        this.cbm = cbm;
    }

    public void setComponentContributionThreshold(double componentContributionThreshold) {
        this.componentContributionThreshold = componentContributionThreshold;
    }

    @Override
    public CBM getModel() {
        return cbm;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        BMDistribution bmDistribution = new BMDistribution(cbm, vector, componentContributionThreshold);
        CBMPredictor cbmPredictor = new CBMPredictor(bmDistribution);
        return cbmPredictor.predictByDynamic();
    }
}
