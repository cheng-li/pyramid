package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 5/21/15.
 */
public class SquaredLossOfExpectation implements Optimizable.ByGradientValue {
    private static final Logger logger = LogManager.getLogger();

    private DataSet dataSet;
    private double[] labels;
    /**
     * format:
     * bias, weights, left output, right output
     */
    private Vector vector;
    private int[] activeFeatures;

    public SquaredLossOfExpectation(DataSet dataSet, double[] labels, int[] activeFeatures) {
        this.dataSet = dataSet;
        this.labels = labels;
        this.vector = new DenseVector(dataSet.getNumFeatures() + 3);
        this.activeFeatures = activeFeatures;

        vector.set(vector.size()-2,0);
        vector.set(vector.size()-1,1);

    }

    public SquaredLossOfExpectation(DataSet dataSet, double[] labels, Vector vector, int[] activeFeatures) {
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


    @Override
    public Vector getGradient() {
        if (logger.isDebugEnabled()){
            logger.debug("calculating gradient");
        }
        Vector gradient = new DenseVector(vector.size());

        Sigmoid sigmoid = new Sigmoid(getWeightsWithoutBias(),getBias());

        double leftValue = getLeftValue();
        double rightValue = getRightValue();
        double[] hx = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> hx[i] = sigmoid.leftProbability(dataSet.getRow(i))
        );

//        if (logger.isDebugEnabled()){
//            logger.debug("hx = "+Arrays.toString(hx));
//        }

        double[] fx = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> fx[i] = hx[i]*leftValue + (1-hx[i])*rightValue
        );

//        if (logger.isDebugEnabled()){
//            logger.debug("fx = "+Arrays.toString(fx));
//        }

        Vector d = new DenseVector(dataSet.getNumDataPoints());
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> d.set(i,(fx[i]-labels[i])*hx[i]*(1-hx[i]))
        );

        //gradient for bias
        gradient.set(0,(leftValue - rightValue)*d.zSum());

        //gradients for all features specified in active features
        Arrays.stream(this.activeFeatures).parallel().forEach(
                j -> gradient.set(j+1,(leftValue - rightValue)*d.dot(dataSet.getColumn(j)))
        );

        //gradient for left value
        double leftGrad = IntStream.range(0,dataSet.getNumDataPoints()).parallel().mapToDouble(
                i->(fx[i]-labels[i])*hx[i]
        ).sum();
        gradient.set(gradient.size()-2,leftGrad);

        //gradient for right value
        double rightGrad = IntStream.range(0,dataSet.getNumDataPoints()).parallel().mapToDouble(
                i->(fx[i]-labels[i])*(1-hx[i])
        ).sum();
        gradient.set(gradient.size()-1,rightGrad);
        if (logger.isDebugEnabled()){
            logger.debug("gradient = "+gradient.toString());
        }
        if (logger.isDebugEnabled()){
            logger.debug("gradient calculation done");
        }
        return gradient;
    }

    private double getValue(){
        Sigmoid sigmoid = new Sigmoid(getWeightsWithoutBias(),getBias());
        double leftValue = getLeftValue();
        double rightValue = getRightValue();
        double[] hx = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> hx[i] = sigmoid.leftProbability(dataSet.getRow(i))
        );

        double[] fx = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> fx[i] = hx[i]*leftValue + (1-hx[i])*rightValue
        );

        double squaredLoss = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> Math.pow(fx[i]-labels[i],2)).sum()*0.5;
        return squaredLoss;
    }

    @Override
    public double getValue(Vector parameters) {
        SquaredLossOfExpectation loss = new SquaredLossOfExpectation(this.dataSet,this.labels,parameters, this.activeFeatures);
        return loss.getValue();
    }

    @Override
    public Vector getParameters() {
        return vector;
    }

    @Override
    public void refresh() {

    }
}
