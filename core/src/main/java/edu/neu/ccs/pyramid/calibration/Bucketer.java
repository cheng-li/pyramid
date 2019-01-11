package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.util.Pair;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class Bucketer {


    /**
     * the last bucket may contain more points
     * @param x group by x
     * @param y
     * @param numPointsInEachBucket
     * @return
     */
    public static Result groupWithEqualSize(double[] x, double[] y, int numPointsInEachBucket){
        int numBuckets = x.length/numPointsInEachBucket;
        List<Pair<Double,Double>> pairs = new ArrayList<>();
        for (int i=0;i<x.length;i++){
            pairs.add(new Pair<>(x[i],y[i]));
        }

        Comparator<Pair<Double,Double>> comparator = Comparator.comparing(Pair::getFirst);
        List<Pair<Double,Double>> sortedPairs = pairs.stream().sorted(comparator).collect(Collectors.toList());
        
        double[] averageX = new double[numBuckets];
        double[] averageY = new double[numBuckets];
        double[] count = new double[numBuckets];

        for (int i=0;i<sortedPairs.size();i++){
            int bucketIndex = i/numPointsInEachBucket;
            if (bucketIndex>=numBuckets){
                bucketIndex = numBuckets - 1;
            }
            averageX[bucketIndex] += sortedPairs.get(i).getFirst();
            averageY[bucketIndex] += sortedPairs.get(i).getSecond();
            count[bucketIndex] += 1;
        }

        for (int a=0;a<averageX.length;a++){
            averageX[a] /= count[a];
            averageY[a] /= count[a];
        }

        Result result = new Result();
        result.averageX = averageX;
        result.averageY = averageY;
        result.count = count;
        return result;

    }

    public static class Result implements Serializable {
        private static final long serialVersionUID = 1L;
        double[] averageX;
        double[] averageY;
        double[] count;


        public double[] getAverageX() {
            return averageX;
        }

        public double[] getAverageY() {
            return averageY;
        }

        public double[] getCount() {
            return count;
        }

        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder("Result{");
            sb.append("averageX=").append(Arrays.toString(averageX)).append("\n");
            sb.append("averageY=").append(Arrays.toString(averageY)).append("\n");
            sb.append("count=").append(Arrays.toString(count)).append("\n");
            sb.append('}');
            return sb.toString();
        }
    }


}
