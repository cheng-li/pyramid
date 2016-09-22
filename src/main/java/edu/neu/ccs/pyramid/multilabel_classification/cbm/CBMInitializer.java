package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.clustering.bm.BMSelector;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.util.MathUtil;


/**
 * Created by chengli on 10/26/15.
 */
public class CBMInitializer {


    public static void initialize(CBM CBM, MultiLabelClfDataSet dataSet, CBMOptimizer optimizer){
        double[][] gamms = BMSelector.selectGammas(dataSet.getNumClasses(),dataSet.getMultiLabels(), CBM.numClusters);
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int k = 0; k< CBM.numClusters; k++){
                optimizer.gammas[i][k] = gamms[i][k];
                optimizer.gammasT[k][i] = gamms[i][k];
            }
        }
        optimizer.mStep();
    }

    public static void randInitialize(CBM CBM, MultiLabelClfDataSet dataSet, CBMOptimizer optimizer) {
        int K = CBM.getNumClusters();

        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            double[] dist = new double[K];
            for (int k=0; k<K; k++) {
                dist[k] = Math.random();
            }
            double sum = MathUtil.arraySum(dist);
            for (int k=0; k<K; k++) {
                double value = dist[k]/sum;
                optimizer.gammas[i][k] = value;
                optimizer.gammasT[k][i] = value;
            }
        }
        optimizer.mStep();
    }

    public static void avgInitialize(CBM CBM, MultiLabelClfDataSet dataSet, CBMOptimizer optimizer) {
        int K = CBM.getNumClusters();
        double avgValue = 1 / (double) K;
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            for (int k=0; k<K; k++) {
                optimizer.gammas[i][k] = avgValue;
                optimizer.gammasT[k][i] = avgValue;
            }
        }
        optimizer.mStep();
    }

}
