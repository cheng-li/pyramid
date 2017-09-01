package edu.neu.ccs.pyramid.classification.l2boost;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.ProbabilityMatrix;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;

import java.util.stream.IntStream;

/**
 * Created by chengli on 10/1/15.
 */
public class L2BoostOptimizer extends GBOptimizer {
    private L2Boost boosting;
    private ProbabilityMatrix probabilityMatrix;
    private double[][] targetDistribution;


    public L2BoostOptimizer(L2Boost boosting, DataSet dataSet, RegressorFactory factory, double[] weights, double[][] targetDistribution) {
        super(boosting, dataSet, factory ,weights);
        this.boosting = boosting;
        this.targetDistribution = targetDistribution;
    }

    public L2BoostOptimizer(L2Boost boosting, DataSet dataSet, double[][] targetDistribution, RegressorFactory factory) {
        this(boosting,dataSet,factory,defaultWeights(dataSet.getNumDataPoints()),targetDistribution);
    }

    public L2BoostOptimizer(L2Boost boosting, ClfDataSet dataSet, RegressorFactory factory) {
        this(boosting,dataSet, DataSetUtil.labelDistribution(dataSet),factory);
    }

    public L2BoostOptimizer(L2Boost boosting, ClfDataSet dataSet) {
        this(boosting,dataSet,defaultFactory());
    }

    @Override
    protected void initializeOthers() {
        this.probabilityMatrix = new ProbabilityMatrix(dataSet.getNumDataPoints(),2);
    }

    @Override
    protected void updateOthers() {
        updateProbMatrix();
    }

    @Override
    protected void addPriors() {

    }

    private void updateProbMatrix(){
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(this::updateProbability);
    }

    /**
     *
     * @param i data index
     */
    private void updateProbability(int i){
        // this is just a number at the moment
        double positiveScore = scoreMatrix.getScoresForData(i)[0];
        double[] scores = new double[2];
        scores[1] = positiveScore;
        double[] probs = boosting.predictClassProbs(scores);
        for (int k=0;k<2;k++){
            this.probabilityMatrix.setProbability(i,k,probs[k]);
        }
    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        // ensemble will always be 0
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->targetDistribution[i][1]-probabilityMatrix.getProbabilitiesForData(i)[1])
                .toArray();
    }


    private static RegressorFactory defaultFactory(){
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new L2BLeafOutputCalculator());
        return regTreeFactory;
    }
}
