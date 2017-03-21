package edu.neu.ccs.pyramid.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.xpath.operations.And;

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
    private double relativeEpsilon = 0.001;
    /**
     * absolute threshold for big change;
     * if operation = and: both have to apply in order to terminate
     * if operation = or: if any applies then terminate
     * for small values, relativeEpsilon is more picky
     * for big values, absoluteEpsilon is more picky
     */
    private double absoluteEpsilon = 0.001;
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
    private Goal goal = Goal.UNDEFINED;
    private boolean allowNaN = false;
    private boolean allowInfinite = false;
    private Operation operation = Operation.AND;
    private int minIterations = 0;

    public Terminator() {
        this.history = new ArrayList<>();
    }

    public void add(double value){
        if (Double.isInfinite(value)&&!allowInfinite){
            throw new RuntimeException("value is infinite");
        }
        if (Double.isNaN(value)&&!allowNaN){
            throw new RuntimeException("value is NaN");
        }
        if (!isMoveValid(value)){
            if (logger.isWarnEnabled()){
                logger.warn("goal = "+goal+", min = "+min+", max = "+max+", current value = "+value);
            }
//            throw new RuntimeException("goal = "+goal+", min = "+min+", max = "+max+", current value = "+value);

        }

        this.history.add(value);
        if (value>max){
            max = value;
        }
        if (value<min){
            min = value;
        }
        if (history.size()>=2){
            double previous = history.get(history.size()-2);
            boolean condition1 = Math.abs(value-previous) <= relativeEpsilon*Math.abs(previous);
            boolean condition2 = Math.abs(value-previous) <= absoluteEpsilon;

            switch (operation){
                case AND:
                    if (condition1&&condition2){
                        stableCounter += 1;
                    } else {
                        stableCounter = 0;
                    }
                    break;
                case OR:
                    if (condition1||condition2){
                        stableCounter += 1;
                    } else {
                        stableCounter = 0;
                    }
                    break;
            }


        }
        if (logger.isDebugEnabled()){
            logger.debug("iteration = "+history.size());
            logger.debug("mode = "+getMode());
            logger.debug("goal = "+getGoal());
            logger.debug("value = "+getLastValue());
            logger.debug("previous value = "+getPreviousValue());
            logger.debug("min value = "+getMinValue());
            logger.debug("max value = "+getMaxValue());
            logger.debug("stable iterations = "+getStableIterations());
            logger.debug("is converged = "+isConverged());
            logger.debug("should terminate = "+shouldTerminate());
        }
    }

    public void setAllowNaN(boolean allowNaN) {
        this.allowNaN = allowNaN;
    }

    public void setAllowInfinite(boolean allowInfinite) {
        this.allowInfinite = allowInfinite;
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
        if (history.size() < minIterations){
            return false;
        }
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

    public Terminator setRelativeEpsilon(double relativeEpsilon) {
        this.relativeEpsilon = relativeEpsilon;
        return this;
    }

    public double getRelativeEpsilon() {
        return relativeEpsilon;
    }

    public double getAbsoluteEpsilon() {
        return absoluteEpsilon;
    }

    public Terminator setAbsoluteEpsilon(double absoluteEpsilon) {
        this.absoluteEpsilon = absoluteEpsilon;
        return this;
    }

    public  Terminator setOperation(Operation operation) {
        this.operation = operation;
        return this;
    }

    public Terminator setMaxStableIterations(int maxStableIterations) {
        this.maxStableIterations = maxStableIterations;
        return this;
    }

    public Terminator setMinIterations(int minIterations) {
        this.minIterations = minIterations;
        return this;
    }

    public int getMaxIteration() {
        return maxIteration;
    }

    public Terminator setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
        return this;
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

    public Terminator setMode(Mode mode) {
        this.mode = mode;
        return this;
    }

    public Goal getGoal() {
        return goal;
    }

    public Terminator setGoal(Goal goal) {
        this.goal = goal;
        return this;
    }


    private boolean isMoveValid(double value){
        boolean valid;
        switch (goal){
            case MINIMIZE:
                valid = (value<=min);
                break;
            case MAXIMIZE:
                valid = (value>=max);
                break;
            case UNDEFINED:
                valid = true;
                break;
            default:
                valid = true;
                break;
        }
        return valid;
    }

    public enum Mode{
        STANDARD,FINISH_MAX_ITER
    }

    public enum Goal{
        MINIMIZE, MAXIMIZE,UNDEFINED
    }

    public enum Operation{
        AND, OR
    }
}
