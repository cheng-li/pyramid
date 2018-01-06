package edu.neu.ccs.pyramid.util;

import org.apache.commons.math3.distribution.BinomialDistribution;

import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/14/14.
 */
public class Sampling {
    /**
     * start with 0
     * @param totalSize
     * @param sampleSize
     * @return
     */
    public static int[] sampleBySize(int totalSize, int sampleSize){
        List<Integer> list = new ArrayList<>(totalSize);
        for (int i=0;i<totalSize;i++){
            list.add(i);
        }
        Collections.shuffle(list);
        int[] sample = new int[sampleSize];
        for (int i=0;i<sampleSize;i++){
            sample[i]=list.get(i);
        }
        return sample;
    }

    /**
     * use ceiling, which means, if there is only one data point,
     * it is guaranteed to be included into the sample
     * @param totalSize
     * @param percentage
     * @return
     */
    public static int[] sampleByPercentage(int totalSize, double percentage){
        int sampleSize = (int)Math.ceil(percentage*totalSize);
        return sampleBySize(totalSize,sampleSize);
    }

    public static List<Integer> stratified(int[] labels, double percentage){
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0;i<labels.length;i++){
            int label= labels[i];
            if (!map.containsKey(label)){
                map.put(label, new ArrayList<>());
            }
            List<Integer> list = map.get(label);
            list.add(i);
            map.put(label,list);
        }

        List<Integer> sample = new ArrayList<>();
        for (Map.Entry<Integer, List<Integer>> entry: map.entrySet()){
            List<Integer> indices = entry.getValue();
            sample.addAll(sampleByPercentage(indices,percentage));
        }

        return sample;
    }

    public static List<Integer> sampleByPercentage(List<Integer> indices, double percentage){
        List<Integer> copy = null;
        try {
            copy = (List) Serialization.deepCopy(indices);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        Collections.shuffle(copy);
        int totalSize = indices.size();
        int sampleSize = (int)Math.ceil(percentage*totalSize);
        return copy.subList(0,sampleSize);

    }

    public static List<Integer> sampleByPercentage(List<Integer> indices, double percentage, long randomSeed){
        List<Integer> copy = null;
        try {
            copy = (List) Serialization.deepCopy(indices);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        Collections.shuffle(copy, new Random(randomSeed));
        int totalSize = indices.size();
        int sampleSize = (int)Math.ceil(percentage*totalSize);
        return copy.subList(0,sampleSize);

    }

    /**
     *
     * @param sampleSize
     * @param start inclusive
     * @param end exclusive
     * @return
     */
    public static IntStream sampleWithReplacement(int sampleSize, int start, int end){
        return new Random().ints(sampleSize, start, end);

    }

    public static IntStream sampleWithReplacement(int sampleSize, List<Integer> indices){
        return sampleWithReplacement(sampleSize,0,indices.size())
                .map(indices::get);
    }

    /**
     *
     * @param min inclusive
     * @param max exclusive
     * @return random number in this range
     */
    public static double doubleUniform(double min, double max){
        return Math.random()*(max-min) + min;

    }

    /**
     * sample uniformly in the log scale
     * @param min
     * @param max
     * @return
     */
    public static double doubleLogUniform(double min, double max){
        if (min<=0){
            throw new IllegalArgumentException("min<=0");
        }

        double minLog = Math.log(min);
        double maxLog = Math.log(max);
        double exp = doubleUniform(minLog, maxLog);
        return Math.exp(exp);
    }

    /**
     *
     * @param min inclusive
     * @param max inclusive
     * @return random int in this range
     */
    public static int intUniform(int min, int max){
        return new Random().nextInt(max - min +1) + min;
    }

    /**
     * sample desired size from first to last with given probabilities
     * if one pass is not enough, rotate
     * assume non-zero probabilities
     */
    public static Set<Integer> rotate(List<Pair<Integer,Double>> probs, int size){
        Set<Integer> res = new HashSet<>();
        if (size==0){
            return res;
        }
        // if not enough candidates, return all
        if (probs.size()<size){
            probs.stream().forEach(pair-> res.add(pair.getFirst()));
            return res;
        }

        boolean next = true;
        while(next){
            for (Pair<Integer,Double> pair: probs){
                int dataIndex = pair.getFirst();
                double prob = pair.getSecond();
                if (res.size()==size){
                    next = false;
                    break;
                }
                if (!res.contains(dataIndex)){
                    BinomialDistribution distribution = new BinomialDistribution(1,prob);
                    int sample = distribution.sample();
                    if (sample==1){
                        res.add(dataIndex);
                    }
                }
            }

        }
        return res;
    }


    /**
     *
     * @param dimension
     * @return a random vector with elements summing up to 1
     */
    public static double[] randomCategoricalDis(int dimension){
        double[] vector = new double[dimension];
        double used = 0;
        for (int i=0;i<dimension;i++){

            double prob;
            if (i==dimension-1){
                prob = 1-used;
            } else {
                prob = Sampling.doubleUniform(0,1-used);
            }

            vector[i]=prob;
            used += prob;
        }
        return vector;
    }

}
