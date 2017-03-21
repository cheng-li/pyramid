package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.GeneralF1Predictor;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import java.util.List;

/**
 * Created by chengli on 3/2/17.
 */
public class CBMSF1Predictor implements PluginPredictor<CBMS> {
    CBMS cbm;

    private List<MultiLabel> support;


    public CBMSF1Predictor(CBMS model) {
        this.cbm = model;
    }

    public CBMSF1Predictor(CBMS cbm, List<MultiLabel> support) {
        this.cbm = cbm;
        this.support = support;
    }


    public void setSupport(List<MultiLabel> support) {
        this.support = support;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        MultiLabel pred = predictBySupport(vector);
        return pred;
    }


    private MultiLabel predictBySupport(Vector vector){
        double[] probs = cbm.predictAssignmentProbs(vector,support);
        return GeneralF1Predictor.predict(cbm.getNumClasses(),support,probs);
    }



    @Override
    public CBMS getModel() {
        return cbm;
    }

}
