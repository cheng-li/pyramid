package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.clustering.bmm.BMMSelector;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;

import java.io.File;


/**
 * Created by chengli on 10/26/15.
 */
public class BMMInitializer {


    public static void initialize(BMMClassifier bmmClassifier, MultiLabelClfDataSet dataSet, BMMOptimizer optimizer){
        double[][] gamms = BMMSelector.selectGammas(dataSet.getNumClasses(),dataSet.getMultiLabels(), bmmClassifier.numClusters);
        optimizer.setParallelism(true);
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int k=0;k<bmmClassifier.numClusters;k++){
                optimizer.gammas[i][k] = gamms[i][k];
                optimizer.gammasT[k][i] = gamms[i][k];
            }
        }
        optimizer.mStep();
    }



}
