package edu.neu.ccs.pyramid.util;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/14/14.
 */
public class MathUtil {

    public static int[] shffuleArray(int[] array) {
        Random rgen = new Random();  // Random number generator

        for (int i=0; i<array.length; i++) {
            int randomPosition = rgen.nextInt(array.length);
            int temp = array[i];
            array[i] = array[randomPosition];
            array[randomPosition] = temp;
        }
        return array;
    }

    /**
     * zeros a array by given size.
     *
     */
    public static double[] zeros(int m) {
        double[] results = new double[m];
        return results;
    }

    /**
     * returns a array including given range.
     * [start, end)
     */
    public static int[] range(int start, int end) {
        int[] results = new int[end-start];
        int index = 0;
        for (int i=start; i<end; i++) {
            results[index++] = i;
        }
        return results;
    }

    /**
     * returns a array including given range.
     * [start, end)
     */
    public static int[] randomRange(int start, int end) {
        Random rgen = new Random();  // Random number generator
        int size = end-start;
        int[] array = new int[size];

        for(int i=0; i< size; i++){
            array[i] = start+i;
        }

        for (int i=0; i<array.length; i++) {
            int randomPosition = rgen.nextInt(array.length);
            int temp = array[i];
            array[i] = array[randomPosition];
            array[randomPosition] = temp;
        }
        return array;
    }

    /**
     * calculate log(exp(x1)+exp(x2)+...)
     * @return
     */
    public static double logSumExp(double[] arr){
        double maxElement = arr[0];
        for (double number: arr){
            if (number > maxElement){
                maxElement = number;
            }
        }
        double sum = 0;
        for (double number: arr){
            sum += Math.exp(number - maxElement);
        }
        return Math.log(sum) + maxElement;
    }

    /**
     * calculate log(exp(x1)+exp(x2)+...)
     * @return
     */
    public static double logSumExp(float[] arr){
        double[] d = new double[arr.length];
        for (int i=0;i<d.length;i++){
            d[i] = arr[i];
        }
        return logSumExp(d);
    }

    /**
     *
     * @param arr
     * @return L1 norm of an array
     */
    public static double l1Norm(double[] arr){
        double norm = 0;
        for (double number: arr){
            norm += Math.abs(number);
        }
        return norm;
    }

    public static double l2Norm(double[] arr){
        double norm = 0;
        for (double number: arr){
            norm += Math.pow(number, 2);
        }
        return Math.sqrt(norm);
    }

    public static double maxNorm(double[] arr){
        double norm = 0;
        for (double number: arr){
            double abs = Math.abs(number);
            if (abs>norm){
                norm = abs;
            }
        }
        return norm;
    }

    /**
     *
     * @param distribution  assume distribution sums up to 1
     * @return
     */
    public static double entropy(double[] distribution){
        double entropy = 0;
        for (double prob: distribution){
            if (prob!=0){
                entropy -= prob*Math.log(prob)/Math.log(2);
            }
        }
        return entropy;
    }

    public static double arraySum(double[] arr){
        double sum = 0;
        for(double num:arr){
            sum += num;
        }
        return sum;
    }

    public static float arraySum(float[] arr){
        float sum = 0;
        for(float num:arr){
            sum += num;
        }
        return sum;
    }

    /**
     * exponentiate scores and normalize
     * @param scores
     * @return probabilities
     */
    public static double[] softmax(double[] scores){
        double[] probVector = new double[scores.length];
        double logDenominator = MathUtil.logSumExp(scores);
        for (int k=0;k<scores.length;k++){
            double logNumerator = scores[k];
            double pro = Math.exp(logNumerator-logDenominator);
            probVector[k]=pro;
        }
        return probVector;
    }


    /**
     * exponentiate scores and normalize and take log
     * @param scores
     * @return probabilities
     */
    public static double[] logSoftmax(double[] scores){
        double[] logProbVector = new double[scores.length];
        double logDenominator = MathUtil.logSumExp(scores);
        for (int k=0;k<scores.length;k++){
            logProbVector[k]=scores[k]- logDenominator;
        }
        return logProbVector;
    }

    /**
     *
     * @param probabilities
     * @return scores that can produce the given probabilities after softmax
     */
    public static double[] inverseSoftMax(double[] probabilities){
        int len = probabilities.length;
        // cannot compute log(0), using log(1E-10) instead
        for (int i=0;i<len;i++){
            if (probabilities[i]==0){
                probabilities[i]=1E-10;
            }
        }
        double[] logs = new double[len];
        for (int i=0;i<len;i++){
            logs[i] = Math.log(probabilities[i]);
        }
        double average = MathUtil.arraySum(logs)/len;
        double[] scores = new double[len];
        for (int i=0;i<len;i++){
            scores[i] = logs[i] - average;
        }
        return scores;
    }

    public static double median(double[] arr){
        DescriptiveStatistics stats = new DescriptiveStatistics(arr);
        return stats.getPercentile(50);
    }

    public static double sign(double d){
        if (d>0){
            return 1;
        } else if (d<0){
            return -1;
        } else {
            return 0;
        }
    }

    // todo: this is a naive implementation
    // see https://en.wikipedia.org/wiki/Weighted_median for a better algorithm
    public static double weightedMedian(double[] scores, double[] weights){
        List<Pair<Double,Double>> pairs = new ArrayList<>();
        for (int i=0;i<scores.length;i++){
            pairs.add(new Pair<>(scores[i], weights[i]));
        }

        Comparator<Pair<Double,Double>> comparator = Comparator.comparing(pair-> pair.getFirst());

        List<Pair<Double,Double>> sorted = pairs.stream().sorted(comparator).collect(Collectors.toList());

        double totalWeight = MathUtil.arraySum(weights);
        double sum = 0;

        for (Pair<Double, Double> aSorted : sorted) {
            sum += aSorted.getSecond();
            if (sum >= totalWeight / 2) {
                return aSorted.getFirst();
            }
        }

        // should not happen
        return Double.NaN;
    }



}
