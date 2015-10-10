package edu.neu.ccs.pyramid.clustering.bmm;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.optimization.*;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 9/12/15.
 */
public class BMMTrainer {
    private static final Logger logger = LogManager.getLogger();
    DataSet dataSet;
    /**
     * gammas[i][k] = probability of data i in cluster k
     */
    double[][] gammas;
    int numClusters;
    BMM bmm;
    Terminator terminator;

    public BMMTrainer(DataSet dataSet,int numClusters) {
        this.numClusters = numClusters;
        this.dataSet = dataSet;
        this.gammas = new double[dataSet.getNumDataPoints()][numClusters];
        this.bmm = new BMM(numClusters,dataSet.getNumFeatures());
        this.terminator = new Terminator();
    }

    public BMM train(){
        while (true){
            iterate();
            if (terminator.shouldTerminate()){
                break;
            }
        }

        return this.bmm;
    }



    void iterate(){
        if (logger.isDebugEnabled()){
            logger.debug("start one EM iteration");
        }
        eStep();
        mStep();
        double objective = getObjective();
        double exactObjective = exactObjective();
        if (logger.isDebugEnabled()){
            logger.debug("finish one EM iteration");
            logger.debug("objective = "+ objective);
            logger.debug("exact objective = "+ exactObjective);
        }
        terminator.add(getObjective());
    }

    private void eStep(){
        if (logger.isDebugEnabled()){
            logger.debug("start E step");
        }
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(this::updateGamma);
        if (logger.isDebugEnabled()){
            logger.debug("finish E step");
            logger.debug("objective = "+ getObjective());
        }
    }

    private void mStep(){
        if (logger.isDebugEnabled()){
            logger.debug("start M step");
        }
        IntStream.range(0,numClusters).parallel().forEach(this::updateCluster);
        if (logger.isDebugEnabled()){
            logger.debug("finish M step");
            logger.debug("objective = "+ getObjective());
        }
    }

    private double getEntropy(){
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::getEntropy).sum();
    }

    private double getEntropy(int i){
        return Entropy.entropy(gammas[i]);
    }

    /**
     *
     * @param k cluster index
     */
    private void updateCluster(int k){
        double nk = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            nk += gammas[i][k];
        }
        Vector average = new DenseVector(dataSet.getNumFeatures());
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            average = average.plus(dataSet.getRow(i).times(gammas[i][k]));
        }
        average = average.divide(nk);
//        if (logger.isDebugEnabled()){
//            logger.debug("average vector = "+average);
//        }
        for (int d=0;d<dataSet.getNumFeatures();d++){
            bmm.distributions[k][d] = new BinomialDistribution(1,average.get(d));
        }

        bmm.mixtureCoefficients[k] = nk/dataSet.getNumDataPoints();
    }


    /**
     * update all gammas
     */
    private void updateGamma(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateGamma);
    }

    /**
     *
     * @param n data point index
     */
    private void updateGamma(int n){
        Vector feature = dataSet.getRow(n);
        int numClusters = bmm.getNumClusters();
        double[] logPis = new double[numClusters];
        for (int k=0;k<numClusters;k++){
            logPis[k] = Math.log(bmm.mixtureCoefficients[k]);
        }
        double[] logClusterConditionalProbs = bmm.clusterConditionalLogProbArr(feature);
        double[] logNumerators = new double[numClusters];
        for (int k=0;k<numClusters;k++){
            logNumerators[k] = logPis[k] + logClusterConditionalProbs[k];
        }
        double logDenominator = MathUtil.logSumExp(logNumerators);
        for (int k=0;k<numClusters;k++){
            gammas[n][k] = Math.exp(logNumerators[k] - logDenominator);
        }
    }


    /**
     * interestingly, the exact objective (not bound) is easy to evaluate
     * use Bishop Eq 9.51 as objective;
     * negative log likelihood, to be minimized
     * @return
     */
    private double exactObjective(){
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel().mapToDouble(this::exactObjective)
                .sum();
    }


    /**
     *
     * @param i data point
     * @return
     */
    private double exactObjective(int i){
        Vector feature = dataSet.getRow(i);
        double[] logPis = new double[numClusters];
        for (int k=0;k<numClusters;k++){
            logPis[k] = Math.log(bmm.mixtureCoefficients[k]);
        }
        double[] logClusterConditionalProbs = bmm.clusterConditionalLogProbArr(feature);
        double[] scores = new double[numClusters];
        for (int k=0;k<numClusters;k++){
            scores[k] = logPis[k] + logClusterConditionalProbs[k];
        }

        return -1*MathUtil.logSumExp(scores);
    }

    private double bernoulliObj(){
        double res =  IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::bernoulliObj).sum();
//        if (logger.isDebugEnabled()){
//            logger.debug("bernoulli objective = "+res);
//        }
        return res;
    }

    private double bernoulliObj(int dataPoint){
        double res = 0;
        for (int k=0;k<bmm.getNumClusters();k++){
            res += bernoulliObj(dataPoint,k);
        }
        return res;
    }

    private double bernoulliObj(int dataPoint, int cluster){
        if (gammas[dataPoint][cluster]<1E-10){
            return 0;
        }
        double sum = 0;
        BinomialDistribution[][] distributions = bmm.distributions;
        int dim = dataSet.getNumFeatures();
        for (int l=0;l<dim;l++){
            double mu = distributions[cluster][l].getProbabilityOfSuccess();
            double value = dataSet.getRow(dataPoint).get(l);
            // unstable if compute directly
            if (value==1){
                if (mu==0){
                    throw new RuntimeException("value=1 and mu=0, gamma nk = "+gammas[dataPoint][cluster]);
                }
                sum += Math.log(mu);

            } else {
                // label == 0
                if (mu==1){
                    throw new RuntimeException("value=0 and mu=1, gamma nk"+gammas[dataPoint][cluster]);
                }
                sum += Math.log(1-mu);
            }
        }
        return -1*gammas[dataPoint][cluster]*sum;
    }


    private double klDivergence(){
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::klDivergence).sum();
    }

    private double klDivergence(int i){
        return KLDivergence.kl(gammas[i],bmm.mixtureCoefficients);
    }

    private double getMStepObjective(){
        return klDivergence() + getEntropy() + bernoulliObj();
    }

    /**
     * the entropy term gets canceled
     * @return
     */
    private double getObjective(){
        return klDivergence() + bernoulliObj();
    }

}
