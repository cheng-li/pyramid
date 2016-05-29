package edu.neu.ccs.pyramid.regression.least_squares_boost;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.optimization.Optimizer;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
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
    private DataSet dataSet;
    private double[] labels;


    public LSBoostTrainer(LSBoost boost, LSBConfig lsbConfig, DataSet dataSet, double[] labels) {
        this.boost = boost;
        this.lsbConfig = lsbConfig;
        this.dataSet = dataSet;
        this.labels = labels;
        boost.featureList = dataSet.getFeatureList();
        int numDataPoints = dataSet.getNumDataPoints();
        this.scores = new double[numDataPoints];
        initScores();
        this.gradients = new double[numDataPoints];
        updateGradients();
    }

    public LSBoostTrainer(LSBoost boost, LSBConfig lsbConfig, RegDataSet dataSet) {
        this(boost, lsbConfig, dataSet, dataSet.getLabels());
    }


    public void iterate(){
        Regressor regressor = fitRegressor();
        addRegressor(regressor,lsbConfig.getLearningRate());
    }

    public void iterate(int numIterations){
        for (int i=0;i<numIterations;i++){
            iterate();
        }
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
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i -> scores[i] += weight*regressor.predict(dataSet.getRow(i))
        );
    }

    void updateGradients(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i -> gradients[i] = labels[i] - scores[i]
        );
    }

    public void addPriorRegressor(){
        Regressor regressor = fitPriorRegressor();
        addRegressor(regressor);
    }

    Regressor fitPriorRegressor(){
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
        return lsbConfig.getRegressorFactory().fit(dataSet,gradients);
    }

}
