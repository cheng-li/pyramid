package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

public class Util {

    private static double brProb(MultiLabel multiLabel, double[] marginals){
        double prod = 1;
        for (int l=0;l<marginals.length;l++){
            if (multiLabel.matchClass(l)){
                prod *= marginals[l];
            } else {
                prod *= 1-marginals[l];
            }
        }
        return prod;
    }
}
