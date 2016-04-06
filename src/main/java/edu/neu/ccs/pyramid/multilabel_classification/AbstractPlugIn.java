package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 4/6/16.
 */
public abstract class AbstractPlugIn implements MultiLabelClassifier {
    protected MultiLabelClassifier model;

    public AbstractPlugIn(MultiLabelClassifier model) {
        this.model = model;
    }

    @Override
    public int getNumClasses() {
        return model.getNumClasses();
    }

    @Override
    abstract public MultiLabel predict(Vector vector);

    @Override
    public FeatureList getFeatureList() {
        return model.getFeatureList();
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return model.getLabelTranslator();
    }
}
