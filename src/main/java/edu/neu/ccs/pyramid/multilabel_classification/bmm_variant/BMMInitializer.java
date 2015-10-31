package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.clustering.bmm.BMMSelector;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;

import java.io.File;


/**
 * Created by chengli on 10/26/15.
 */
public class BMMInitializer {
    public static void initialize(BMMClassifier bmmClassifier, MultiLabelClfDataSet dataSet, double softmaxVar, double binaryLogVar){
        double[][] gamms = BMMSelector.selectGammas(dataSet.getNumClasses(),dataSet.getMultiLabels(), bmmClassifier.numClusters);
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,dataSet,softmaxVar,binaryLogVar);
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int k=0;k<bmmClassifier.numClusters;k++){
                optimizer.gammas[i][k] = gamms[i][k];
                optimizer.gammasT[k][i] = gamms[i][k];
            }
        }
        optimizer.mStep();
    }

    public static void initialize(BMMClassifier bmmClassifier, MultiLabelClfDataSet dataSet, double softmaxVar, double binaryLogVar, File modelFile) throws Exception {

        edu.neu.ccs.pyramid.multilabel_classification.bmm.BMMOptimizer optimizer1 = edu.neu.ccs.pyramid.multilabel_classification.bmm.BMMOptimizer.deserialize(modelFile);
        double[][] gamms = optimizer1.gammas;
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,dataSet,softmaxVar,binaryLogVar);
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int k=0;k<bmmClassifier.numClusters;k++){
                optimizer.gammas[i][k] = gamms[i][k];
                optimizer.gammasT[k][i] = gamms[i][k];
            }
        }
        optimizer.mStep();
    }
}
