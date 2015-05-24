package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;

/**
 * Created by chengli on 5/21/15.
 */
public class ProbRegStumpTrainer {
    private DataSet dataSet;
    private SquaredLoss squaredLoss;
    private LBFGS lbfgs;

    public ProbRegStumpTrainer(DataSet dataSet, double[] labels) {
        this.dataSet = dataSet;
        this.squaredLoss = new SquaredLoss(dataSet,labels);
        this.lbfgs = new LBFGS(squaredLoss);
    }

    /**
     * set parameters here
     * @return
     */
    public LBFGS getLbfgs() {
        return lbfgs;
    }

    public ProbRegStump train(){
        lbfgs.optimize();

//        GradientDescent gradientDescent = new GradientDescent(squaredLoss,1);
//        for (int i=0;i<100;i++){
//            gradientDescent.update();
//        }

        GatingFunction gatingFunction = new Sigmoid(squaredLoss.getWeightsWithoutBias(),squaredLoss.getBias());


        ProbRegStump probRegStump = new ProbRegStump();
        probRegStump.gatingFunction = gatingFunction;
        probRegStump.leftOutput = squaredLoss.getLeftValue();
        probRegStump.rightOutput = squaredLoss.getRightValue();
        probRegStump.featureList = dataSet.getFeatureList();

        return probRegStump;
    }
}
