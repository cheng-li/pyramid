package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.clustering.bm.BMSelector;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;

/**
 * Created by chengli on 11/17/16.
 */
public class CBMSInitializer {
    public static void initialize(CBMS cbms, MultiLabelClfDataSet dataSet, CBMSOptimizer optimizer){
        double[][] gamms = BMSelector.selectGammas(dataSet.getNumClasses(),dataSet.getMultiLabels(), cbms.getNumComponents());
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int k = 0; k< cbms.getNumComponents(); k++){
                optimizer.gammas[i][k] = gamms[i][k];
            }
        }
        System.out.println("performing M step");
        optimizer.mStep();
    }
}
