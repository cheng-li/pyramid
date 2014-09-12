package edu.neu.ccs.pyramid.util;

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
        Collections.shuffle(indices);
        int totalSize = indices.size();
        int sampleSize = (int)Math.ceil(percentage*totalSize);
        return indices.subList(0,sampleSize);

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

}
