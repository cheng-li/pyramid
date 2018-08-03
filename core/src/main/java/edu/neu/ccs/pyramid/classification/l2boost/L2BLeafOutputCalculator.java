package edu.neu.ccs.pyramid.classification.l2boost;

import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputCalculator;

/**
 * Created by chengli on 10/2/15.
 */
public class L2BLeafOutputCalculator implements LeafOutputCalculator{
    @Override
    public double getLeafOutput(double[] probabilities, double[] labels) {
        double numerator = 0;
        double denominator = 0;
        for (int i=0;i<probabilities.length;i++) {
            double label = labels[i];
            numerator += label*probabilities[i];
            denominator += Math.abs(label) * (1 - Math.abs(label))*probabilities[i];
        }
        double out;
        if (denominator == 0) {
            out = 0;
        } else {
            out = numerator / (2 * denominator);
        }
        //protection from numerically unstable issue
        //todo does the threshold matter?
        if (out>1){
            out=1;
        }
        if (out<-1){
            out=-1;
        }
        if (Double.isNaN(out)) {
            throw new RuntimeException("leaf value is NaN");
        }
        if (Double.isInfinite(out)){
            throw new RuntimeException("leaf value is Infinite");
        }
        return out;
    }

    @Override
    public void setParallel(boolean parallel) {

    }
}
