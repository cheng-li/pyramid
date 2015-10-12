package edu.neu.ccs.pyramid.eval;

/**
 * Created by chengli on 11/11/14.
 */
public class KLDivergence {
    public static double kl(double[] targetDistribution, double[] estimatedDistribution){
        double r = 0;
        for (int i=0;i<targetDistribution.length;i++){
            // if ==0, don't change sum
            if (targetDistribution[i]!=0){
                r += targetDistribution[i]* Math.log(targetDistribution[i] / estimatedDistribution[i]);
            }
        }
        return r;
    }

    public static double klGivenPLogQ(double[] targetDistribution, double[] logEstimatedDistribution){
        double r = 0;
        for (int i=0;i<targetDistribution.length;i++){
            // if ==0, don't change sum
            if (targetDistribution[i]!=0){
                r += targetDistribution[i]* (Math.log(targetDistribution[i])-logEstimatedDistribution[i]);
            }
        }
        return r;
    }
}
