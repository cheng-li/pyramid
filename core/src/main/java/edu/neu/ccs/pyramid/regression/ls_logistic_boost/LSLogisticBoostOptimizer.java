package edu.neu.ccs.pyramid.regression.ls_logistic_boost;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Sigmoid;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.stream.IntStream;

public class LSLogisticBoostOptimizer extends GBOptimizer {
    private static final Logger logger = LogManager.getLogger();
    private double[] labels;

    /**
     *
     * @param boosting
     * @param dataSet
     * @param factory
     * @param weights
     * @param labels between [0,1]
     */
    public LSLogisticBoostOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory, double[] weights, double[] labels) {
        super(boosting, dataSet, factory, weights);
        this.labels = labels;
    }

    public LSLogisticBoostOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory, double[] labels) {
        super(boosting, dataSet, factory);
        this.labels = labels;
    }


    public LSLogisticBoostOptimizer(GradientBoosting boosting, RegDataSet dataSet, RegressorFactory factory) {
        this(boosting,  dataSet, factory, dataSet.getLabels());
    }

    @Override
    protected void addPriors() {
        double average = IntStream.range(0,dataSet.getNumDataPoints()).parallel().mapToDouble(i-> labels[i]*weights[i]).average().getAsDouble();
        if (average>=1||average<=0){
            throw new RuntimeException("input average >=1 or <=0");
        }
        double inverse = MathUtil.inverseSigmoid(average);
        Regressor constant = new ConstantRegressor(inverse);
        boosting.getEnsemble(0).add(constant);
    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().
                mapToDouble(this::gradientForInstance).toArray();
    }

    private double gradientForInstance(int i){
        double p = Sigmoid.sigmoid(scoreMatrix.getScoresForData(i)[0]);
        //todo
        double g = -2*(p-labels[i])*(1-p)*p;
//        if (Math.abs(g)<1){
//            g = MathUtil.sign(g)*1;
//        }
        return g;
//        return -2*(p-labels[i])*(1-p)*p;
//        return -2*(p-labels[i]);
    }

    @Override
    protected void initializeOthers() {
        return;
    }

    @Override
    protected void updateOthers() {
        return;
    }
}
