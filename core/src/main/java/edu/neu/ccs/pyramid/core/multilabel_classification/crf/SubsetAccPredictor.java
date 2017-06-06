package edu.neu.ccs.pyramid.core.multilabel_classification.crf;

import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.multilabel_classification.PluginPredictor;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/19/16.
 */
public class SubsetAccPredictor implements PluginPredictor<CMLCRF> {
    private static final long serialVersionUID = 1L;
    private CMLCRF crf;

    public SubsetAccPredictor(CMLCRF crf) {
        this.crf = crf;
    }

    @Override
    public CMLCRF getModel() {
        return crf;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        return crf.predict(vector);
    }
}
