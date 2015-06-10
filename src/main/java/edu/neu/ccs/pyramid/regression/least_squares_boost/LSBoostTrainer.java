package edu.neu.ccs.pyramid.regression.least_squares_boost;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizer;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStumpTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.*;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 6/3/15.
 */
public class LSBoostTrainer {
    private static final Logger logger = LogManager.getLogger();
    private double[] scores;
    private double[] gradients;
    private LSBoost boost;
    private LSBConfig lsbConfig;

    public LSBoostTrainer(LSBoost boost, LSBConfig lsbConfig) {
        this.boost = boost;
        this.lsbConfig = lsbConfig;
        RegDataSet dataSet = lsbConfig.getDataSet();
        boost.featureList = dataSet.getFeatureList();
        int numDataPoints = dataSet.getNumDataPoints();
        this.scores = new double[numDataPoints];
        initScores();
        this.gradients = new double[numDataPoints];
        updateGradients();
    }


    public void iterate(){
        Regressor regressor = fitRegressor();
        addRegressor(regressor,lsbConfig.getLearningRate());
    }

    // the prior regressor should not be shrunk
    public void addRegressor(Regressor regressor, double weight){
        boost.regressors.add(regressor);
        boost.weights.add(weight);
        updateScores(regressor,weight);
        updateGradients();
    }


    public void addRegressor(Regressor regressor){
        addRegressor(regressor,1.0);
    }

    public double[] getGradients() {
        return gradients;
    }

    void updateScores(Regressor regressor, double weight){
        DataSet dataSet = lsbConfig.getDataSet();
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i -> scores[i] += weight*regressor.predict(dataSet.getRow(i))
        );
    }

    void updateGradients(){
        RegDataSet dataSet = lsbConfig.getDataSet();
        double[] labels = dataSet.getLabels();
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i -> gradients[i] = labels[i] - scores[i]
        );
    }

    public void addPriorRegressor(){
        Regressor regressor = fitPriorRegressor();
        addRegressor(regressor);
    }

    Regressor fitPriorRegressor(){
        RegDataSet dataSet = lsbConfig.getDataSet();
        double[] labels = dataSet.getLabels();
        double ave = Arrays.stream(labels).average().getAsDouble();
        return new ConstantRegressor(ave);
    }

    private void initScores(){
        for (int i=0;i<boost.regressors.size();i++){
            Regressor regressor = boost.regressors.get(i);
            double weight = boost.weights.get(i);
            updateScores(regressor,weight);
        }
    }
    
    private Regressor fitRegressor(){
        List<Pair<Regressor,Double>> competitors = new ArrayList<>();

        if (lsbConfig.considerHardTree()){
            LeafOutputCalculator leafOutputCalculator = new AverageOutputCalculator(gradients);
            
            RegTreeConfig regTreeConfig = new RegTreeConfig();
            regTreeConfig.setMaxNumLeaves(this.lsbConfig.getNumLeaves());
            regTreeConfig.setMinDataPerLeaf(this.lsbConfig.getMinDataPerLeaf());
            regTreeConfig.setActiveDataPoints(this.lsbConfig.getActiveDataPoints());
            regTreeConfig.setActiveFeatures(this.lsbConfig.getActiveFeatures());
            regTreeConfig.setNumSplitIntervals(this.lsbConfig.getNumSplitIntervals());
            regTreeConfig.setRandomLevel(this.lsbConfig.getRandomLevel());

            RegressionTree regressionTree = RegTreeTrainer.fit(regTreeConfig,
                    this.lsbConfig.getDataSet(),
                    gradients,
                    leafOutputCalculator);
            // use un-shrunk one to calculate mse
            double mse = MSE.mse(gradients, regressionTree.predict(lsbConfig.getDataSet()));
            System.out.println("hard tree mse = "+mse);
            
            competitors.add(new Pair<>(regressionTree, mse));
        }


        if (lsbConfig.considerExpectationTree()){
            ProbRegStumpTrainer expectationTrainer = ProbRegStumpTrainer.getBuilder()
                    .setDataSet(lsbConfig.getDataSet())
                    .setLabels(gradients)
                    .setFeatureType(ProbRegStumpTrainer.FeatureType.FOLLOW_HARD_TREE_FEATURE)
                    .setLossType(ProbRegStumpTrainer.LossType.SquaredLossOfExpectation)
                    //todo
                    .setOptimizerType(ProbRegStumpTrainer.OptimizerType.LBFGS)
                    .build();

            Optimizer optimizer = expectationTrainer.getOptimizer();
            optimizer.setCheckConvergence(lsbConfig.softTreeEarlyStop());
            optimizer.setMaxIteration(100);



            ProbRegStump expectationTree = expectationTrainer.train();

            double mse = MSE.mse(gradients, expectationTree.predict(lsbConfig.getDataSet()));
            System.out.println("expectation tree mse = "+mse);
            competitors.add(new Pair<>(expectationTree, mse));
        }

        if (lsbConfig.considerProbabilisticTree()){
            ProbRegStumpTrainer probabilisticTrainer = ProbRegStumpTrainer.getBuilder()
                    .setDataSet(lsbConfig.getDataSet())
                    .setLabels(gradients)
                    .setFeatureType(ProbRegStumpTrainer.FeatureType.FOLLOW_HARD_TREE_FEATURE)
                    .setLossType(ProbRegStumpTrainer.LossType.ExpectationOfSquaredLoss)
                    .build();

            ProbRegStump probabilisticTree = probabilisticTrainer.train();


            double mse = MSE.mse(gradients,probabilisticTree.predict(lsbConfig.getDataSet()));
            System.out.println("probabilistic tree mse = "+mse);

            competitors.add(new Pair<>(probabilisticTree, mse));
        }

        if (competitors.isEmpty()){
            throw new RuntimeException("no regressor is considered");
        }

        Comparator<Pair<Regressor,Double>> comparator = Comparator.comparing(Pair::getSecond);

        List<Pair<Regressor,Double>> sorted = competitors.stream().sorted(comparator).collect(Collectors.toList());

        return sorted.get(0).getFirst();
    }

}
