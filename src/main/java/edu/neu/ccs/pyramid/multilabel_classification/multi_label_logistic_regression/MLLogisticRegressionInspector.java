package edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.feature.FeatureUtility;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import org.apache.mahout.math.Vector;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 2/5/15.
 */
public class MLLogisticRegressionInspector {

    public static TopFeatures topFeatures(MLLogisticRegression logisticRegression,
                                                   int classIndex, int limit){
        FeatureList featureList = logisticRegression.getFeatureList();
        Vector weights = logisticRegression.getWeights().getWeightsWithoutBiasForClass(classIndex);
        Comparator<FeatureUtility> comparator = Comparator.comparing(FeatureUtility::getUtility);
        List<Feature> list = IntStream.range(0, weights.size())
                .mapToObj(i -> new FeatureUtility(featureList.get(i)).setUtility(weights.get(i)))
                .filter(featureUtility -> featureUtility.getUtility()>0)
                .sorted(comparator.reversed())
                .map(FeatureUtility::getFeature)
                .limit(limit)
                .collect(Collectors.toList());
        TopFeatures topFeatures = new TopFeatures();
        topFeatures.setTopFeatures(list);
        topFeatures.setClassIndex(classIndex);
        LabelTranslator labelTranslator = logisticRegression.getLabelTranslator();
        topFeatures.setClassName(labelTranslator.toExtLabel(classIndex));
        return topFeatures;
    }
}
