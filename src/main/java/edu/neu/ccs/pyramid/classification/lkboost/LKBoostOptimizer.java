package edu.neu.ccs.pyramid.classification.lkboost;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.neu.ccs.pyramid.regression.regression_tree.*;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/14/14.
 */
public class LKBoostOptimizer extends GBOptimizer {
    private static final Logger logger = LogManager.getLogger();
    private ProbabilityMatrix probabilityMatrix;
    private double[][] targetDistribution;
    private LKBoost boosting;
    private int numClasses;


    public LKBoostOptimizer(LKBoost boosting, DataSet dataSet, RegressorFactory factory, double[] weights, double[][] targetDistribution) {
        super(boosting, dataSet, factory ,weights);
        this.boosting = boosting;
        this.targetDistribution = targetDistribution;
        this.numClasses = boosting.getNumClasses();
    }

    public LKBoostOptimizer(LKBoost boosting, ClfDataSet dataSet, RegressorFactory factory, double[] weights) {
        this(boosting,dataSet, factory, weights,DataSetUtil.labelDistribution(dataSet));
    }

    public LKBoostOptimizer(LKBoost boosting, ClfDataSet dataSet, RegressorFactory factory) {
        this(boosting,dataSet, factory, defaultWeights(dataSet.getNumDataPoints()),DataSetUtil.labelDistribution(dataSet));
    }

    public LKBoostOptimizer(LKBoost boosting, ClfDataSet dataSet, double[] weights) {
        this(boosting,dataSet,defaultFactory(dataSet.getNumClasses()),weights);
    }

    public LKBoostOptimizer(LKBoost boosting, ClfDataSet dataSet) {
        this(boosting,dataSet,defaultFactory(dataSet.getNumClasses()));
    }

    @Override
    protected void initializeOthers() {
        this.probabilityMatrix = new ProbabilityMatrix(dataSet.getNumDataPoints(),numClasses);
    }

    @Override
    protected void updateOthers() {
        updateProbabilityMatrix();
    }

    @Override
    protected void addPriors() {
        //todo
    }

//    public void addPriorRegressors(){
//        PriorProbClassifier priorProbClassifier = new PriorProbClassifier(this.lkTreeBoost.getNumClasses());
//        priorProbClassifier.fit(this.lktbConfig.getDataSet());
//        double[] probs = priorProbClassifier.getClassProbs();
//        double average = Arrays.stream(probs).map(Math::log).average().getAsDouble();
//        List<Regressor> regressors = new ArrayList<>();
//        for (int k=0;k<this.lkTreeBoost.getNumClasses();k++){
//            double score = Math.log(probs[k] - average);
//            Regressor constant = new ConstantRegressor(score);
//            regressors.add(constant);
//        }
//        addRegressors(regressors);
//    }

    public GradientMatrix getGradientMatrix() {
        return gradientMatrix;
    }

    public ProbabilityMatrix getProbabilityMatrix() {
        return probabilityMatrix;
    }


    //======================== PRIVATE ===============================================


    /**
     * parallel by classes
     * calculate gradient vectors for all classes, store them
     */
    protected void updateGradientMatrix(){
        int numDataPoints = this.dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(this::updateClassGradients);
    }

    private void updateClassGradients(int dataPoint){
        int numClasses = this.boosting.getNumClasses();
        double[] probs = this.probabilityMatrix.getProbabilitiesForData(dataPoint);
        for (int k=0;k<numClasses;k++){
            double gradient;
            gradient = targetDistribution[dataPoint][k] - probs[k];
            this.gradientMatrix.setGradient(dataPoint,k,gradient);
        }
    }

    /**
     * use scoreMatrix to update probabilities
     * numerically unstable if calculated directly
     * probability = exp(log(numerator)-log(denominator))
     */
    private void updateClassProb(int i){
        int numClasses = this.boosting.getNumClasses();
        double[] scores = scoreMatrix.getScoresForData(i);

        double logDenominator = MathUtil.logSumExp(scores);
//        if (logger.isDebugEnabled()){
//            logger.debug("logDenominator for data point "+i+" with scores  = "+ Arrays.toString(scores)
//                    +" ="+logDenominator+", label = "+lktbConfig.getDataSet().getLabels()[i]);
//        }
        for (int k=0;k<numClasses;k++){
            double logNumerator = scores[k];
            double pro = Math.exp(logNumerator-logDenominator);
            this.probabilityMatrix.setProbability(i,k,pro);
            if (Double.isNaN(pro)){
                throw new RuntimeException("pro=NaN, logNumerator = "
                        +logNumerator+", logDenominator="+logDenominator+
                        ", scores = "+Arrays.toString(scores));

            }
        }
    }

    /**
     * use scoreMatrix to update probabilities
     * parallel by data
     */
    private void updateProbabilityMatrix(){
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0,numDataPoints).parallel()
                .forEach(this::updateClassProb);
    }

    private static RegressorFactory defaultFactory(int numClasses){
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(numClasses));
        return regTreeFactory;
    }

}
