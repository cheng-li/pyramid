package edu.neu.ccs.pyramid.regression.regression_tree;

/**
 * probabilistic sum of squared error
 * Created by chengli on 10/16/14.
 */
class ProbabilisticSSE {

    static double sse(double[] labels, double[] probs){
        double sum = 0;
        double mean = mean(labels,probs);
        for (int i=0;i<labels.length;i++){
            double error = labels[i] - mean;
            sum += error*error*probs[i];
        }
        return sum;
    }

    /**
     * weighed average, weights are probs
     * @param labels
     * @param probs
     * @return
     */
    static double mean(double[] labels, double[] probs){
        double nominator = 0;
        double denominator = 0;
        for (int i=0;i<labels.length;i++){
            nominator += labels[i]*probs[i];
            denominator += probs[i];
        }
        return nominator/denominator;
    }


}
