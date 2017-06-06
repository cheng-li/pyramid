package edu.neu.ccs.pyramid.core.multilabel_classification;

import edu.neu.ccs.pyramid.core.feature.FeatureList;
import edu.neu.ccs.pyramid.core.dataset.LabelTranslator;

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
