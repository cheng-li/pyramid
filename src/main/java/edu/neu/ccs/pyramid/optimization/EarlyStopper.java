package edu.neu.ccs.pyramid.optimization;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 10/29/16.
 */
public class EarlyStopper {
    private List<Double> values;
    private List<Integer> iterations;
    private double bestValue;
    private int bestIteration = -1;
    private Goal goal;
    private int patience = 5;
    private int patienceUses = 0;
    private boolean shouldStop = false;
    private int minimumIterations = 20;

    public EarlyStopper(Goal goal, int patience) {
        this.goal = goal;
        this.patience = patience;
        if (goal==Goal.MAXIMIZE){
            bestValue = Double.NEGATIVE_INFINITY;
        } else {
            bestValue = Double.POSITIVE_INFINITY;
        }
        this.values = new ArrayList<>();
        this.iterations = new ArrayList<>();
    }

    public void setMinimumIterations(int minimumIterations) {
        this.minimumIterations = minimumIterations;
    }

    public void add(int iteration, double value){

        if (!shouldStop){
            values.add(value);
            iterations.add(iteration);
            boolean better = isBetter(value);
            if (better){
                updateWithProgress(iteration, value);
            } else {
                updateWithoutProgress(iteration);
            }
        }
    }

    public int getBestIteration() {
        return bestIteration;
    }

    public boolean shouldStop() {
        return shouldStop;
    }

    public enum Goal{
        MAXIMIZE, MINIMIZE
    }


    private boolean isBetter(double value){
        if (goal==Goal.MAXIMIZE && value > bestValue){
            return true;
        }

        if (goal==Goal.MINIMIZE && value < bestValue){
            return true;
        }

        return false;
    }


    private void updateWithProgress(int iteration, double value){
        bestIteration = iteration;
        bestValue = value;
        patienceUses = 0;
    }

    private void updateWithoutProgress(int iteration){
        patienceUses += 1;
        if (patienceUses>=patience && iteration>=minimumIterations){
            shouldStop = true;
        }
    }

    public String history(){
        StringBuilder stringBuilder = new StringBuilder();
        for (int i=0;i<iterations.size();i++){
            stringBuilder.append(iterations.get(i)).append(":");
            stringBuilder.append(values.get(i));
            if (iterations.get(i)==bestIteration){
                stringBuilder.append("[best]");
            }
            if (i!=iterations.size()-1){
                stringBuilder.append(", ");
            }
        }
        return stringBuilder.toString();
    }






}
