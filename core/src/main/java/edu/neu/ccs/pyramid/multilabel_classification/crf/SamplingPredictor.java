package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.SamplingPrediction;
import org.apache.mahout.math.Vector;

import java.util.List;

/**
 * Created by chengli on 10/16/16.
 */
public class SamplingPredictor implements PluginPredictor<CMLCRF> {
    private CMLCRF cmlcrf;

    public SamplingPredictor(CMLCRF cmlcrf) {
        this.cmlcrf = cmlcrf;
    }

    @Override
    public CMLCRF getModel() {
        return cmlcrf;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        double[] probabilities = cmlcrf.predictCombinationProbs(vector);
        List<MultiLabel> support = cmlcrf.getSupportCombinations();
        return SamplingPrediction.predict(probabilities, support);
    }
}
