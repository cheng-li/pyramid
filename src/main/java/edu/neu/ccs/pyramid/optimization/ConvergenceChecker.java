package edu.neu.ccs.pyramid.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 9/4/15.
 */
public class ConvergenceChecker {
    private static final Logger logger = LogManager.getLogger();
    private double min = Double.POSITIVE_INFINITY;
    private double max = Double.NEGATIVE_INFINITY;
    private List<Double> history;
    private int counter = 0;
    /**
     * relative threshold for big change
     */
    private double epsilon = 0.05;
    /**
     * if no big change in maxStableIterations, regard as converge
     */
    private int maxStableIterations = 5;
    private int maxIteration = 10000;

    public ConvergenceChecker() {
        this.history = new ArrayList<>();
    }

    public void add(double value){
        this.history.add(value);
        if (value>max){
            max = value;
        }
        if (value<min){
            min = value;
        }
        if (history.size()>=2){
            double previous = history.get(history.size()-2);
            if (Math.abs((value-previous)/previous)<=epsilon){
                counter += 1;
            } else {
                counter = 0;
            }
        }
        if (logger.isDebugEnabled()){
            logger.debug("iteration = "+history.size());
            logger.debug("last value = "+getLastValue());
            logger.debug("min value = "+getMinValue());
            logger.debug("max value = "+getMaxValue());
            logger.debug("stable iterations = "+getStableIterations());
            logger.debug("is converged = "+isConverged());
        }
    }

    public double getMaxValue(){
        return this.max;
    }

    public double getMinValue(){
        return this.min;
    }

    public int getStableIterations() {
        return counter;
    }

    public boolean isConverged(){
        return (counter >= maxStableIterations)||(history.size() >= maxIteration);
    }

    public int getNumIterations(){
        return history.size();
    }

    public List<Double> getHistory() {
        return history;
    }

    public double getLastValue(){
        return history.get(history.size()-1);
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setMaxStableIterations(int maxStableIterations) {
        this.maxStableIterations = maxStableIterations;
    }

    public int getMaxIteration() {
        return maxIteration;
    }

    public void setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
    }
}
