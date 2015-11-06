package edu.neu.ccs.pyramid.eval;

import org.apache.commons.math3.util.FastMath;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;

/**
 * Created by chengli on 11/11/14.
 */
public class KLDivergence {
    private static final Logger logger = LogManager.getLogger();

    public static double kl(double[] trueDistribution, double[] estimatedDistribution){
        double r = 0;
        for (int i=0;i<trueDistribution.length;i++){
            if (trueDistribution[i]==0){
                r += 0;
            } else {
                r += trueDistribution[i]* (Math.log(trueDistribution[i])-Math.log(estimatedDistribution[i]));
            }
        }
        if (Double.isInfinite(r)&&logger.isDebugEnabled()){
            logger.debug("true distribution = "+ Arrays.toString(trueDistribution));
            logger.debug("estimated distribution = "+ Arrays.toString(estimatedDistribution));
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
