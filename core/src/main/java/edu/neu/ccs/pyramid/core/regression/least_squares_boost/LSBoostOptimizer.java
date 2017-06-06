package edu.neu.ccs.pyramid.core.regression.least_squares_boost;

import edu.neu.ccs.pyramid.core.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.core.dataset.DataSet;
import edu.neu.ccs.pyramid.core.dataset.RegDataSet;
import edu.neu.ccs.pyramid.core.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.core.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.core.regression.Regressor;
import edu.neu.ccs.pyramid.core.regression.RegressorFactory;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.stream.IntStream;

/**
 * Created by chengli on 6/3/15.
 */
public class LSBoostOptimizer extends GBOptimizer{
    private static final Logger logger = LogManager.getLogger();
    private double[] labels;

    public LSBoostOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory, double[] weights, double[] labels) {
        super(boosting, dataSet, factory, weights);
        this.labels = labels;
    }

    public LSBoostOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory, double[] labels) {
        super(boosting, dataSet, factory);
        this.labels = labels;
    }


    public LSBoostOptimizer(GradientBoosting boosting, RegDataSet dataSet, RegressorFactory factory) {
        this(boosting,  dataSet, factory, dataSet.getLabels());
    }

    @Override
    protected void addPriors() {
        double average = IntStream.range(0,dataSet.getNumDataPoints()).parallel().mapToDouble(i-> labels[i]*weights[i]).average().getAsDouble();
        Regressor constant = new ConstantRegressor(average);
        boosting.getEnsemble(0).add(constant);
    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().
                mapToDouble(i->labels[i]-scoreMatrix.getScoresForData(i)[0]).toArray();
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
