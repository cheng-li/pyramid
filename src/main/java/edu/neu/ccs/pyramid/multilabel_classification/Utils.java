package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

import java.util.List;

/**
 * Created by chengli on 5/17/17.
 */
public class Utils {

    public static double[] marginals(List<MultiLabel> combinations, double[] combinationProbs, int numLabels){
        double[] marginals = new double[numLabels];
        for (int i=0;i<combinations.size();i++){
            MultiLabel y = combinations.get(i);
            double p = combinationProbs[i];
            for (int l: y.getMatchedLabels()){
                marginals[l] += p;
            }
        }
        return marginals;
    }
}
