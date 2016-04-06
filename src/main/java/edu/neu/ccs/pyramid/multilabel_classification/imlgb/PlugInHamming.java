package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.AbstractPlugIn;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import org.apache.mahout.math.Vector;

/**
 * optimal for Hamming Loss
 * Created by chengli on 4/6/16.
 */
public class PlugInHamming extends AbstractPlugIn{
    private static final long serialVersionUID = 1L;
    private IMLGradientBoosting imlGradientBoosting;

    public PlugInHamming(IMLGradientBoosting imlGradientBoosting) {
        super(imlGradientBoosting);
        this.imlGradientBoosting = imlGradientBoosting;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        MultiLabel prediction = new MultiLabel();
        for (int k=0;k<getNumClasses();k++){
            double score = imlGradientBoosting.predictClassScore(vector, k);
            if (score > 0){
                prediction.addLabel(k);
            }
        }
        return prediction;
    }

}
