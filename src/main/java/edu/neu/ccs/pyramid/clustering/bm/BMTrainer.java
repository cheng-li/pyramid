package edu.neu.ccs.pyramid.clustering.bm;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.optimization.*;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by chengli on 9/12/15.
 */
public class BMTrainer {
    private static final Logger logger = LogManager.getLogger();
    DataSet dataSet;
    /**
     * gammas[i][k] = probability of data i in cluster k
     */
    double[][] gammas;
    int numClusters;
    BM bm;
//    Terminator terminator;
    int numIterations=200;

    public BMTrainer(DataSet dataSet, int numClusters, long randomSeed) {
        this.numClusters = numClusters;
        this.dataSet = dataSet;
        this.gammas = new double[dataSet.getNumDataPoints()][numClusters];
        this.bm = new BM(numClusters,dataSet.getNumFeatures(), randomSeed);

//        this.terminator = new Terminator();
//        this.terminator.setAbsoluteEpsilon(0.1);
//        this.terminator.setMaxIteration(200);
    }

    public BM train(){
        //todo change back
        for (int i=0;i<numIterations;i++){
            iterate();
        }
//        while (true){
//            iterate();
//            if (terminator.shouldTerminate()){
//                break;
//            }
//        }

        return this.bm;
    }

    public BM getBm() {
        return bm;
    }

    public double[][] getGammas() {
        return gammas;
    }

    public void iterate(){
        if (logger.isDebugEnabled()){
            logger.debug("start one EM iteration");
        }
        eStep();
        mStep();

//        double objective = getObjective();
//        if (logger.isDebugEnabled()){
//            logger.debug("finish one EM iteration");
//            logger.debug("objective = "+ objective);
////            double exactObjective = exactObjective();
////            logger.debug("exact objective = "+ exactObjective);
//        }
//        terminator.add(getObjective());

    }

    public void eStep(){
        if (logger.isDebugEnabled()){
            logger.debug("start E step");
        }
        updateGamma();
        if (logger.isDebugEnabled()){
            logger.debug("finish E step");
            logger.debug("objective = "+ getObjective());
        }
    }

    public  void mStep(){
        if (logger.isDebugEnabled()){
            logger.debug("start M step");
        }
        IntStream.range(0,numClusters).forEach(this::updateCluster);
        bm.updateLogClusterConditioinalForEmpty();
        if (logger.isDebugEnabled()){
            logger.debug("finish M step");
            logger.debug("objective = "+ getObjective());
        }
    }


    /**
     *
     * @param k cluster index
     */
    private void updateCluster(int k){

        final double effectiveTotal = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i-> gammas[i][k]).sum();

        IntStream.range(0, dataSet.getNumFeatures()).parallel()
                .forEach(d-> {
                    double sum = weightedSum(k, d);
                    double average = sum/effectiveTotal;
                    // it may happen that average = 1.0000000000000022, for numerical reasons
                    if (average>=1){
                        average = 0.9999;
                    }
                    bm.distributions[k][d] = new BernoulliDistribution(average);
                });

        bm.mixtureCoefficients[k] = effectiveTotal/dataSet.getNumDataPoints();
        bm.logMixtureCoefficients[k] = Math.log(bm.mixtureCoefficients[k]);
    }

    private double weightedSum(int clusterIndex, int dimensionIndex){
        Vector column = dataSet.getColumn(dimensionIndex);
        double sum = 0;
        for (Vector.Element nonzero: column.nonZeroes()){
            int i = nonzero.index();
            sum += gammas[i][clusterIndex];
        }
        return sum;
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
        int numClusters = bm.getNumClusters();

        double[] logClusterConditionalProbs = bm.clusterConditionalLogProbArr(feature);
        double[] logNumerators = new double[numClusters];
        for (int k=0;k<numClusters;k++){
            logNumerators[k] = bm.logMixtureCoefficients[k] + logClusterConditionalProbs[k];
        }
        double logDenominator = MathUtil.logSumExp(logNumerators);
        for (int k=0;k<numClusters;k++){
            gammas[n][k] = Math.exp(logNumerators[k] - logDenominator);
        }
    }


    /**
     * the exact objective (not bound) is easy to evaluate
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
        return -1*bm.logProbability(dataSet.getRow(i));
    }


    public double getObjective(){
        return exactObjective();
    }

    // the part below computes the bound; since the exact obj is easy to compute, there is no need to compute the bound

//    private double getMStepObjective(){
//        return klDivergence() + getEntropy() + bernoulliObj();
//    }
//
//    private double bernoulliObj(){
//        double res =  IntStream.range(0, dataSet.getNumDataPoints()).parallel()
//                .mapToDouble(this::bernoulliObj).sum();
////        if (logger.isDebugEnabled()){
////            logger.debug("bernoulli objective = "+res);
////        }
//        return res;
//    }
//
//    private double bernoulliObj(int dataPoint){
//        double res = 0;
//        for (int k = 0; k< bm.getNumClusters(); k++){
//            res += bernoulliObj(dataPoint,k);
//        }
//        return res;
//    }
//
//    @Deprecated
//    private double bernoulliObj(int dataPoint, int cluster){
//        if (gammas[dataPoint][cluster]<1E-10){
//            return 0;
//        }
//        double sum = 0;
//        BernoulliDistribution[][] distributions = bm.distributions;
//        int dim = dataSet.getNumFeatures();
//        for (int l=0;l<dim;l++){
//            double mu = distributions[cluster][l].getP();
//            //todo
//            double value = dataSet.getRow(dataPoint).get(l);
//            // unstable if compute directly
//            if (value==1){
//                if (mu==0){
//                    throw new RuntimeException("value=1 and mu=0, gamma nk = "+gammas[dataPoint][cluster]);
//                }
//                sum += Math.log(mu);
//
//            } else {
//                // label == 0
//                if (mu==1){
//                    throw new RuntimeException("value=0 and mu=1, gamma nk"+gammas[dataPoint][cluster]);
//                }
//                sum += Math.log(1-mu);
//            }
//        }
//        return -1*gammas[dataPoint][cluster]*sum;
//    }
//
//
//    private double klDivergence(){
//        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
//                .mapToDouble(this::klDivergence).sum();
//    }
//
//    private double klDivergence(int i){
//        return KLDivergence.kl(gammas[i], bm.mixtureCoefficients);
//    }
//
//    private double getEntropy(){
//        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
//                .mapToDouble(this::getEntropy).sum();
//    }
//
//    private double getEntropy(int i){
//        return Entropy.entropy(gammas[i]);
//    }

}
