package edu.neu.ccs.pyramid.visualizer;

/**
 * Created by shikhar on 7/4/17.
 */
public class PredictedRanking {
    int classIndex;
    String className;
    double prob;
    String type;

    @Override
    public String toString() {
        return "{" + "classIndex: " + classIndex
                + ",className: "+className
                +", prob: "+prob
                + ", type: "+type+ "}";
    }
}