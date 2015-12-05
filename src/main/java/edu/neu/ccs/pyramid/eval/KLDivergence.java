package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.DataSet;
import org.apache.commons.math3.util.FastMath;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

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

    public static double kl(Classifier.ProbabilityEstimator estimator, Vector vector, double[] targetDistribution){
        double[] logEstimation = estimator.predictLogClassProbs(vector);
        return KLDivergence.klGivenPLogQ(targetDistribution,logEstimation);
    }

    public static double kl(Classifier.ProbabilityEstimator estimator, DataSet dataSet,
                     double[][] targetDistributions, double[] weights) {
        double sum = 0.0;
        for(int n=0; n<dataSet.getNumDataPoints(); n++) {
            sum += weights[n] * kl(estimator, dataSet.getRow(n), targetDistributions[n]);
        }
        return sum;
    }

    public static double kl(Classifier.ProbabilityEstimator estimator, DataSet dataSet,
                     double[][] targetDistributions) {
        double[] weights = new double[dataSet.getNumDataPoints()];
        Arrays.fill(weights,1.0);
        return kl(estimator,dataSet,targetDistributions,weights);
    }
}
