package edu.neu.ccs.pyramid.core.regression;

/**
 * Created by chengli on 2/28/15.
 */
public class ConstantRule implements Rule{
    private double score;

    public ConstantRule(double score) {
        this.score = score;
    }

    public double getScore(){
        return score;
    }


}
