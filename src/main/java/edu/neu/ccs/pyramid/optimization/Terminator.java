package edu.neu.ccs.pyramid.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

/**
 * terminate optimization loops
 * Created by chengli on 9/4/15.
 */
public class Terminator {
    private static final Logger logger = LogManager.getLogger();
    private double min = Double.POSITIVE_INFINITY;
    private double max = Double.NEGATIVE_INFINITY;
    private List<Double> history;
    private int stableCounter = 0;
    /**
     * relative threshold for big change
     */
    private double epsilon = 0.01;
    /**
     * if no big change in maxStableIterations, regard as converge
     */
    private int maxStableIterations = 5;
    /**
     * terminate if maxStableIterations is reached
     */
    private int maxIteration = 10000;
    private boolean forceTerminated = false;
    private Mode mode = Mode.STANDARD;

    public Terminator() {
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
                stableCounter += 1;
            } else {
                stableCounter = 0;
            }
        }
        if (logger.isDebugEnabled()){
            logger.debug("iteration = "+history.size());
            logger.debug("mode = "+getMode());
            logger.debug("value = "+getLastValue());
            logger.debug("previous value = "+getPreviousValue());
            logger.debug("min value = "+getMinValue());
            logger.debug("max value = "+getMaxValue());
            logger.debug("stable iterations = "+getStableIterations());
            logger.debug("is converged = "+isConverged());
            logger.debug("should terminate = "+shouldTerminate());
        }
    }

    public double getMaxValue(){
        return this.max;
    }

    public double getMinValue(){
        return this.min;
    }

    public int getStableIterations() {
        return stableCounter;
    }

    public boolean shouldTerminate(){
        boolean ter = false;
        switch (mode){
            case STANDARD:
                ter = isConverged()||(history.size() >= maxIteration)|| forceTerminated;
                break;
            case FINISH_MAX_ITER:
                ter = (history.size() >= maxIteration);
                break;
        }
        return ter;
    }

    public boolean isConverged(){
        return (stableCounter >= maxStableIterations);
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

    public double getPreviousValue(){
        if (history.size()<2){
            return Double.NaN;
        } else {
            return history.get(history.size()-2);
        }
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

    /**
     * user decide to terminate
     */
    public void forceTerminate(){
        this.forceTerminated = true;
    }

    public Mode getMode() {
        return mode;
    }

    public void setMode(Mode mode) {
        this.mode = mode;
    }

    public enum Mode{
        STANDARD,FINISH_MAX_ITER
    }
}
