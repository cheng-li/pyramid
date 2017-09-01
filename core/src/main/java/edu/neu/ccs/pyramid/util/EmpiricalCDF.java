package edu.neu.ccs.pyramid.util;

import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 3/26/15.
 */
public class EmpiricalCDF {
    private double min;
    private double max;
    /**
     * internally, keep one more interval for (-\infty, min]
     * # intervals = # bins +1
     */
    private int numBins;
    private int totalCount;
    private int[] counts;
    private int[] cumulativeCounts;
    private double[] thresholds;
    private double[] probabilities;

    /**
     * if global min and global max are provided, use them, in order to match thresholds
     * @param list
     * @param globalMin
     * @param globalMax
     * @param numBins
     */
    public EmpiricalCDF(List<Double> list, double globalMin, double globalMax, int numBins) {
        this.numBins = numBins;
        int numIntervals = numBins +1;
        this.min = globalMin;
        this.max = globalMax;
        double length = (max - min)/numBins;
        this.thresholds = new double[numIntervals];
        this.probabilities = new double[numIntervals];
        this.counts = new int[numIntervals];
        for (Double value: list){
            int intervalIndex = getIntervalIndex(value,min,length,numIntervals);
            counts[intervalIndex] += 1;
        }
        this.cumulativeCounts = new int[numIntervals];
        for (int i=0;i<cumulativeCounts.length;i++){
            cumulativeCounts[i] += counts[i];
            if (i!=0){
                cumulativeCounts[i] += cumulativeCounts[i-1];
            }
        }
        for (int i=0;i<numIntervals;i++){
            thresholds[i] = min + length*i;
        }
        totalCount = list.size();
        for (int i=0;i<numIntervals;i++){
            probabilities[i] = (double)cumulativeCounts[i]/totalCount;
        }
    }

    /**
     * if no global min and max are provided, local min and max are inferred from data
     * @param list
     * @param numBins
     */
    public EmpiricalCDF(List<Double> list, int numBins) {
        this(list,list.stream().mapToDouble(i->i).min().getAsDouble(),
                list.stream().mapToDouble(i->i).max().getAsDouble(),
                numBins);
    }

    public EmpiricalCDF(List<Double> list) {
        this(list,100);
    }

    private static int getIntervalIndex(double featureValue, double minFeature, double intervalLength, int numIntervals){
        int ceil = (int)Math.ceil((featureValue-minFeature)/intervalLength);
        //this should not happen in theory
        //add this to handle round error
        if (ceil>numIntervals-1){
            ceil=numIntervals-1;
        }
        int intervalIndex;
        if (ceil<=0){
            intervalIndex = 0;
        } else {
            intervalIndex = ceil;
        }
        return intervalIndex;
    }

    public static double distance(EmpiricalCDF cdf1, EmpiricalCDF cdf2){
        if (cdf1.numBins!=cdf2.numBins || cdf1.min!=cdf2.min ||cdf1.max!=cdf2.max){
            throw new IllegalArgumentException("cdf1.numBins!=cdf2.numBins || cdf1.min!=cdf2.min ||cdf1.max!=cdf2.max");
        }
        return IntStream.range(0,cdf1.numBins+1).mapToDouble(i -> Math.abs(cdf1.probabilities[i]-cdf2.probabilities[i]))
                .max().getAsDouble();

    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("EmpiricalCDF{");
        sb.append("min=").append(min);
        sb.append(", max=").append(max);
        sb.append(", numBins=").append(numBins);
        sb.append(", totalCount=").append(totalCount);
        sb.append(", counts=").append(Arrays.toString(counts));
        sb.append(", cumulativeCounts=").append(Arrays.toString(cumulativeCounts));
        sb.append(", thresholds=").append(Arrays.toString(thresholds));
        sb.append(", probabilities=").append(Arrays.toString(probabilities));
        sb.append('}');
        return sb.toString();
    }
}
