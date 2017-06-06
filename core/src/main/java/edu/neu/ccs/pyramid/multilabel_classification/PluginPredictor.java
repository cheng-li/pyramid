package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.FeatureList;

/**
 * a plugin predictor on top of a multi-label classifier E
 * Created by chengli on 5/29/16.
 */
public interface PluginPredictor<E extends MultiLabelClassifier> extends MultiLabelClassifier {
    E getModel();

    default int  getNumClasses(){
        return getModel().getNumClasses();
    }


    default FeatureList getFeatureList(){
        return getModel().getFeatureList();
    }


    default LabelTranslator getLabelTranslator(){
        return getModel().getLabelTranslator();
    }
}
