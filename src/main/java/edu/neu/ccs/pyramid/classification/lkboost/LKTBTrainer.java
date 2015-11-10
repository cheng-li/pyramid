package edu.neu.ccs.pyramid.classification.lkboost;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.classification.l2boost.L2BLeafOutputCalculator;
import edu.neu.ccs.pyramid.classification.l2boost.L2Boost;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.regression.regression_tree.*;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/14/14.
 */
public class LKTBTrainer extends GBOptimizer {
    private static final Logger logger = LogManager.getLogger();
    private ProbabilityMatrix probabilityMatrix;
    private double[][] targetDistribution;
    private LKTreeBoost boosting;
    private int numClasses;


    public LKTBTrainer(LKTreeBoost boosting, DataSet dataSet, double[][] targetDistribution, RegressorFactory factory) {
        super(boosting, dataSet, factory);
        this.boosting = boosting;
        this.targetDistribution = targetDistribution;
        this.numClasses = boosting.getNumClasses();
    }

    public LKTBTrainer(LKTreeBoost boosting, ClfDataSet dataSet, RegressorFactory factory) {
        this(boosting,dataSet, DataSetUtil.labelDistribution(dataSet),factory);
    }


    public LKTBTrainer(LKTreeBoost boosting, ClfDataSet dataSet) {
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


//todo
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
        regTreeFactory.setLeafOutputCalculator(new LKTBLeafOutputCalculator(numClasses));
        return regTreeFactory;
    }

}
