package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.GeneralF1Predictor;
import org.apache.mahout.math.Vector;

import java.util.List;

/**
 * Created by Rainicy on 5/7/16.
 */
public class InstanceF1Predictor implements PluginPredictor<CMLCRF> {
    private CMLCRF cmlcrf;
    private int numClasses;

    public InstanceF1Predictor(CMLCRF model) {
        this.cmlcrf = model;
        this.numClasses = model.getNumClasses();
    }

    @Override
    public MultiLabel predict(Vector vector) {
        List<MultiLabel> supports = cmlcrf.getSupportCombinations();
        double[] probs = cmlcrf.predictCombinationProbs(vector);
        return GeneralF1Predictor.predict(numClasses, supports, probs);
    }

    public void showPredictBySupport(Vector vector, MultiLabel truth){
        System.out.println("support procedure");
        List<MultiLabel> support = cmlcrf.getSupportCombinations();
        double[] probs = cmlcrf.predictCombinationProbs(vector);

        MultiLabel prediction =  GeneralF1Predictor.predict(cmlcrf.getNumClasses(),support,probs);
        System.out.println(GeneralF1Predictor.showSupportPrediction(support,probs, truth, prediction, cmlcrf.getNumClasses()));
    }

    @Override
    public CMLCRF getModel() {
        return cmlcrf;
    }
}
