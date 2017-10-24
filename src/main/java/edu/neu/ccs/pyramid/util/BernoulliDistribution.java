package edu.neu.ccs.pyramid.util;

import java.io.Serializable;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by chengli on 10/11/15.
 */
public class BernoulliDistribution implements Serializable{
    private double p;
    // can be -infinity
    private double logP;
    // can be -infinity
    private double logOneMinusP;

    public BernoulliDistribution(double p) {
        if (p<0||p>1){
            throw new IllegalArgumentException("p should be within 0 and 1. Given p = "+p);
        }
        this.p = p;
        this.logP = Math.log(p);
        this.logOneMinusP = Math.log(1-p);
    }

    public double probability(double value){
        boolean condition = value==0|| value==1;
        if (!condition){
            throw new IllegalArgumentException("value should be 1 or 0");
        }
        if (value==1){
            return p;
        } else{
            return 1-p;
        }
    }

    public double logProbability(int value){
        boolean condition = value==0|| value==1;
        if (!condition){
            throw new IllegalArgumentException("value should be 1 or 0");
        }
        if (value==1){
            return logP;
        } else{
            return logOneMinusP;
        }
    }

    /**
     * return log probability without checking the input.
     * @param value
     * @return
     */
    public double fastLogProbability(int value) {
        return (value==1) ? logP : logOneMinusP;
    }

    public int sample(){
        double d = ThreadLocalRandom.current().nextDouble();
        if (d<p){
            return 1;
        } else {
            return 0;
        }
    }

    public double getP() {
        return p;
    }
}
