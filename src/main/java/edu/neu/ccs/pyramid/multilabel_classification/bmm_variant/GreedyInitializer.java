package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;

/**
 * Created by chengli on 11/3/15.
 */
public class GreedyInitializer {
    // format [#data][#cluster]
    double[][] gammasAllClusters;
    // format [#cluster][#data]
    double[][] gammasAllClustersT;
    MultiLabelClfDataSet dataSet;



    void train(int k){
        double[] gammas = gammasAllClustersT[k];

    }
}
