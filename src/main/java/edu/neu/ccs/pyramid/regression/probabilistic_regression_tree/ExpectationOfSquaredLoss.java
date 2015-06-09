package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 5/27/15.
 */
public class ExpectationOfSquaredLoss implements Optimizable.ByGradientValue{
    private static final Logger logger = LogManager.getLogger();

    private DataSet dataSet;
    private double[] labels;
    /**
     * format:
     * bias, weights, left output, right output
     */
    private Vector vector;
    private int[] activeFeatures;

    private double value;
    private Vector gradient;
    private boolean isGradientCacheValid;
    private boolean isValueCacheValid;

    public ExpectationOfSquaredLoss(DataSet dataSet, double[] labels, int[] activeFeatures) {
        this.dataSet = dataSet;
        this.labels = labels;
        this.vector = new DenseVector(dataSet.getNumFeatures() + 3);
        this.activeFeatures = activeFeatures;

        vector.set(vector.size()-2,0);
        vector.set(vector.size()-1,1);

    }



    public ExpectationOfSquaredLoss(DataSet dataSet, double[] labels, Vector vector, int[] activeFeatures) {
        this.dataSet = dataSet;
        this.labels = labels;
        this.vector = vector;
        this.activeFeatures = activeFeatures;
    }



    Vector getWeightsWithoutBias(){
        return vector.viewPart(1,vector.size()-3);
    }

    double getBias(){
        return vector.get(0);
    }

    double getLeftValue(){
        return vector.get(vector.size()-2);
    }

    double getRightValue(){
        return vector.get(vector.size()-1);
    }



    private void updateGradient() {
        if (logger.isDebugEnabled()){
            logger.debug("calculating gradient");
        }
        Vector gradient = new DenseVector(vector.size());

        Sigmoid sigmoid = new Sigmoid(getWeightsWithoutBias(),getBias());

        double leftValue = getLeftValue();
        double rightValue = getRightValue();
        double[] hx = new double[dataSet.getNumDataPoints()];
        IntStream.range(0, dataSet.getNumDataPoints()).parallel().forEach(
                i-> hx[i] = sigmoid.leftProbability(dataSet.getRow(i))
        );

//        if (logger.isDebugEnabled()){
//            logger.debug("hx = "+Arrays.toString(hx));
//        }



        Vector d = new DenseVector(dataSet.getNumDataPoints());
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i->
                { double di = (Math.pow(leftValue-labels[i],2)-Math.pow(rightValue-labels[i],2))*hx[i]*(1-hx[i]);
                    d.set(i,di);}
        );

        //gradient for bias
        gradient.set(0,d.zSum());

        //gradients for all features specified in active features
        Arrays.stream(this.activeFeatures).parallel().forEach(
                j -> gradient.set(j+1,d.dot(dataSet.getColumn(j)))
        );

        //gradient for left value
        double leftGrad = IntStream.range(0,dataSet.getNumDataPoints()).parallel().mapToDouble(
                i -> 2 * (leftValue - labels[i]) * hx[i]
        ).sum();
        gradient.set(gradient.size()-2,leftGrad);

        //gradient for right value
        double rightGrad = IntStream.range(0,dataSet.getNumDataPoints()).parallel().mapToDouble(
                i -> 2 * (rightValue - labels[i]) * (1 - hx[i])
        ).sum();
        gradient.set(gradient.size()-1,rightGrad);
        if (logger.isDebugEnabled()){
            logger.debug("gradient = "+gradient.toString());
        }
        if (logger.isDebugEnabled()){
            logger.debug("gradient calculation done");
        }
        this.gradient = gradient;
    }

    private void updateValue(){
        Sigmoid sigmoid = new Sigmoid(getWeightsWithoutBias(),getBias());
        double leftValue = getLeftValue();
        double rightValue = getRightValue();
        double[] hx = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> hx[i] = sigmoid.leftProbability(dataSet.getRow(i))
        );

        double loss = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> hx[i] * Math.pow(leftValue - labels[i], 2) + (1 - hx[i]) * Math.pow(rightValue - labels[i], 2))
                .sum();
        this.value = loss;
    }



    @Override
    public Vector getParameters() {
        return vector;
    }

    @Override
    public void setParameters(Vector parameters) {
        this.vector = parameters;
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;
    }

    public double getValue(){
        if (isValueCacheValid){
            return this.value;
        }
        updateValue();
        this.isValueCacheValid = true;
        return this.value;
    }


    public Vector getGradient(){
        if (isGradientCacheValid){
            return this.gradient;
        }
        updateGradient();
        this.isGradientCacheValid = true;
        return this.gradient;
    }
}
