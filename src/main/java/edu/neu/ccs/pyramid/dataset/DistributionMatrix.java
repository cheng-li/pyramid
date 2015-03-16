package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.util.MathUtil;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 3/15/15.
 */
public class DistributionMatrix {
    private int numDataPoints;
    private int numClasses;
    private double[][] dataClass;
    private double[][] classData;

    public DistributionMatrix(int numDataPoints, int numClasses) {
        this.numDataPoints = numDataPoints;
        this.numClasses = numClasses;
        this.dataClass = new double[numDataPoints][numClasses];
        this.classData = new double[numClasses][numDataPoints];
    }

    public void setProbability(int dataPointIndex, int classIndex, double prob){
        this.dataClass[dataPointIndex][classIndex] = prob;
        this.classData[classIndex][dataPointIndex] = prob;
    }

    public double[] getProbsForData(int dataPointIndex){
        return dataClass[dataPointIndex];
    }

    public double[] getProbsForClass(int classIndex){
        return classData[classIndex];
    }

    public void normalize(){
        double sum = Arrays.stream(dataClass).parallel().mapToDouble(MathUtil::arraySum)
                .sum();
        IntStream.range(0,numDataPoints).parallel().forEach(i -> IntStream.range(0, numClasses)
                        .forEach(k -> {
                            double prob = dataClass[i][k] / sum;
                            setProbability(i, k, prob);
                        })
        );
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("DistributionMatrix{");
        sb.append("dataClass=").append(Arrays.deepToString(dataClass));
        sb.append(", classData=").append(Arrays.deepToString(classData));
        sb.append('}');
        return sb.toString();
    }
}
