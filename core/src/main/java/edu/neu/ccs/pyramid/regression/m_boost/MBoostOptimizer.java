package edu.neu.ccs.pyramid.regression.m_boost;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/8/16.
 */
public class MBoostOptimizer extends GBOptimizer {
    private static final Logger logger = LogManager.getLogger();
    private double[] labels;
    // 1-alpha = break down point
    private double alpha = 0.9;

    public MBoostOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory, double[] weights, double[] labels, double alpha) {
        super(boosting, dataSet, factory, weights);
        this.labels = labels;
        this.alpha = alpha;
    }

    public MBoostOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory, double[] labels, double alpha) {
        super(boosting, dataSet, factory);
        this.labels = labels;
        this.alpha = alpha;
    }


    public MBoostOptimizer(GradientBoosting boosting, RegDataSet dataSet, RegressorFactory factory, double alpha) {
        this(boosting,  dataSet, factory, dataSet.getLabels(), alpha);
    }

    @Override
    protected void addPriors() {
        double median = MathUtil.weightedMedian(labels, weights);
        Regressor constant = new ConstantRegressor(median);
        boosting.getEnsemble(0).add(constant);
    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        double[] residual  = IntStream.range(0, dataSet.getNumDataPoints()).parallel().
                mapToDouble(i->labels[i]-scoreMatrix.getScoresForData(i)[0]).toArray();
        double[] absResidual = Arrays.stream(residual).map(Math::abs).toArray();
        DescriptiveStatistics statistics = new DescriptiveStatistics(absResidual);
        double threshold = statistics.getPercentile(alpha*100);
        double[] gradient = new double[residual.length];
        for (int i=0;i<gradient.length;i++){
            if (absResidual[i]<=threshold){
                gradient[i] = residual[i];
            } else {
                gradient[i]= threshold*MathUtil.sign(residual[i]);
            }
        }

        // todo leave output
        return gradient;

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
