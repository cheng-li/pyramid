package edu.neu.ccs.pyramid.core.dataset;

/**
 * Created by chengli on 2/3/15.
 */
public class ScoreMatrix {
    // num data * num classes
    private float[][] m;


    public ScoreMatrix(int numDataPoints, int numClasses) {
        this.m = new float[numDataPoints][numClasses];
    }

    public void setScore(int dataPointIndex, int classIndex, double score){
        this.m[dataPointIndex][classIndex] = (float)score;
    }

    public float[] getScoresForData(int dataPointIndex){
        return m[dataPointIndex];
    }


    public void increment(int dataPointIndex, int classIndex, double increment){
        this.m[dataPointIndex][classIndex] += increment;
    }
}
