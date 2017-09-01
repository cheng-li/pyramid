package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputCalculator;

/**
 * Created by chengli on 10/2/15.
 */
public class IMLGBLeafOutputCalculator implements LeafOutputCalculator{
    private int numClasses;

    public IMLGBLeafOutputCalculator(int numClasses) {
        this.numClasses = numClasses;
    }

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
            out = ((numClasses - 1) * numerator) / (numClasses * denominator);
        }
        //protection from numerically unstable issue
        if (out>2){
            out=2;
        }
        if (out<-2){
            out=-2;
        }
        if (Double.isNaN(out)) {
            throw new RuntimeException("leaf value is NaN");
        }
        if (Double.isInfinite(out)){
            throw new RuntimeException("leaf value is Infinite");
        }
        return out;
    }
}
