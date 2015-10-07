package edu.neu.ccs.pyramid.multilabel_classification.bmm;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 10/7/15.
 */
public class BMM implements MultiLabelClassifier {
    int numClasses;
    int numClusters;
    /**
     * format:[cluster][label]
     */
    BinomialDistribution[][] distributions;
    LogisticRegression logisticRegression;

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    @Override
    // todo bingyu
    public MultiLabel predict(Vector vector) {
        return null;
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return null;
    }
}
