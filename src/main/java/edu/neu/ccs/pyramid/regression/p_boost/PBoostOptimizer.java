package edu.neu.ccs.pyramid.regression.p_boost;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/8/16.
 */
public class PBoostOptimizer extends GBOptimizer {
    private static final Logger logger = LogManager.getLogger();
    private double[] labels;

    public PBoostOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory, double[] weights, double[] labels) {
        super(boosting, dataSet, factory, weights);
        this.labels = labels;
    }

    public PBoostOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory, double[] labels) {
        super(boosting, dataSet, factory);
        this.labels = labels;
    }


    public PBoostOptimizer(GradientBoosting boosting, RegDataSet dataSet, RegressorFactory factory) {
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
        int n = dataSet.getNumDataPoints();
        double labelAve = MathUtil.arraySum(labels)/n;

        double[] pred = IntStream.range(0, n).mapToDouble(i->scoreMatrix.getScoresForData(i)[0]).toArray();
        double predAve = MathUtil.arraySum(pred)/n;
        double[] labelDev = IntStream.range(0, n).mapToDouble(i->labels[i]-labelAve).toArray();
        double[] predDev = IntStream.range(0, n).mapToDouble(i->pred[i]-predAve).toArray();
        double labelDevAve = MathUtil.arraySum(labelDev)/n;
        double predDevAve = MathUtil.arraySum(predDev)/n;

        double product = IntStream.range(0, n).mapToDouble(i->predDev[i]*labelDev[i]).sum();

        double sigmaSquqre = IntStream.range(0, n).mapToDouble(i->Math.pow(predDev[i],2)).sum();

        if (sigmaSquqre==0){
            sigmaSquqre= 1;
        }

        double sigma = Math.sqrt(sigmaSquqre);

        //todo second order; to fix vannishing gradient
        double[] gradient = new double[n];
        for (int i=0;i<n;i++){
            double g = (labelDev[i]-labelDevAve)*sigma - 1/sigma*(product*(predDev[i]-predDevAve));
            g = g/sigmaSquqre;
            gradient[i] = g;
        }
//        System.out.println("-----------------------------");
//        System.out.println(Arrays.toString(gradient));
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
