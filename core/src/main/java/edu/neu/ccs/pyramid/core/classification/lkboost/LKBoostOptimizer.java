package edu.neu.ccs.pyramid.core.classification.lkboost;

import edu.neu.ccs.pyramid.core.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.DataSet;
import edu.neu.ccs.pyramid.core.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.core.dataset.ProbabilityMatrix;
import edu.neu.ccs.pyramid.core.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.core.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.core.regression.Regressor;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.core.util.MathUtil;
import edu.neu.ccs.pyramid.core.regression.RegressorFactory;
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

    public LKBoostOptimizer(LKBoost boosting, DataSet dataSet, RegressorFactory factory, double[][] targetDistribution) {
        this(boosting,dataSet,factory,defaultWeights(dataSet.getNumDataPoints()),targetDistribution);
    }

    public LKBoostOptimizer(LKBoost boosting, ClfDataSet dataSet, RegressorFactory factory, double[] weights) {
        this(boosting,dataSet, factory, weights, DataSetUtil.labelDistribution(dataSet));
        this.boosting.labelTranslator = dataSet.getLabelTranslator();
    }

    public LKBoostOptimizer(LKBoost boosting, ClfDataSet dataSet, RegressorFactory factory) {
        this(boosting,dataSet, factory, defaultWeights(dataSet.getNumDataPoints()),DataSetUtil.labelDistribution(dataSet));
        this.boosting.labelTranslator = dataSet.getLabelTranslator();
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
        PriorProbClassifier priorProbClassifier = new PriorProbClassifier(numClasses);
        priorProbClassifier.fit(dataSet, targetDistribution, weights);
        double[] probs = priorProbClassifier.getClassProbs();
        double[] scores = MathUtil.inverseSoftMax(probs);
        // weaken the priors
        for (int i=0;i<scores.length;i++){
            if (scores[i]>5){
                scores[i]=5;
            }
            if (scores[i]<-5){
                scores[i]=-5;
            }
        }
        for (int k=0;k<numClasses;k++){
            Regressor constant = new ConstantRegressor(scores[k]);
            boosting.getEnsemble(k).add(constant);
        }
    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().mapToDouble(i->gradient(ensembleIndex, i)).toArray();
    }

    //======================== PRIVATE ===============================================


    private double gradient(int ensembleIndex, int dataPoint){
        double prob = probabilityMatrix.getProbabilitiesForData(dataPoint)[ensembleIndex];
        return targetDistribution[dataPoint][ensembleIndex] - prob;
    }


    /**
     * use scoreMatrix to update probabilities
     * numerically unstable if calculated directly
     * probability = exp(log(numerator)-log(denominator))
     */
    private void updateClassProb(int i){
        int numClasses = this.boosting.getNumClasses();
        float[] scores = scoreMatrix.getScoresForData(i);

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
