package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 2/3/15.
 */
public class ProbabilityMatrix {
    private int numDataPoints;
    private int numClasses;
    private double[][] dataClass;
    private double[][] classData;

    public ProbabilityMatrix(int numDataPoints, int numClasses) {
        this.numDataPoints = numDataPoints;
        this.numClasses = numClasses;
        this.dataClass = new double[numDataPoints][numClasses];
        this.classData = new double[numClasses][numDataPoints];
    }

    public void setProbability(int dataPointIndex, int classIndex, double probability){
        this.dataClass[dataPointIndex][classIndex] = probability;
        this.classData[classIndex][dataPointIndex] = probability;
    }

    public double[] getProbabilitiesForData(int dataPointIndex){
        return dataClass[dataPointIndex];
    }

    public double[] getProbabilitiesForClass(int classIndex){
        return classData[classIndex];
    }

    public void increment(int dataPointIndex, int classIndex, double increment){
        this.dataClass[dataPointIndex][classIndex] += increment;
        this.classData[classIndex][dataPointIndex] += increment;
    }


}

