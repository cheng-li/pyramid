package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.clustering.bmm.BMMSelector;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.util.MathUtil;

import java.util.Arrays;

/**
 * Created by chengli on 11/10/15.
 */
public class MixBoostInitializer {
    public static void initialize(BMMClassifier bmmClassifier, MultiLabelClfDataSet dataSet){
        double[][] gamms = BMMSelector.selectGammas(dataSet.getNumClasses(), dataSet.getMultiLabels(), bmmClassifier.numClusters);
        MixBoostOptimizer optimizer = new MixBoostOptimizer(bmmClassifier,dataSet);
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int k=0;k<bmmClassifier.numClusters;k++){
                optimizer.gammas[i][k] = gamms[i][k];
                optimizer.gammasT[k][i] = gamms[i][k];
            }
        }
        optimizer.mStep();
    }
}
