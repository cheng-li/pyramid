package edu.neu.ccs.pyramid.util;

import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/20/14.
 */
public class ArgSort {
    public static int[] argSortAscending(double[] arr){
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        return IntStream.range(0,arr.length)
                .mapToObj(i-> new Pair<>(i,arr[i]))
                .sorted(comparator)
                .mapToInt(Pair::getFirst).toArray();
    }

    public static int[] argSortAscending(List<Double> arr){
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        return IntStream.range(0,arr.size())
                .mapToObj(i-> new Pair<>(i,arr.get(i)))
                .sorted(comparator)
                .mapToInt(Pair::getFirst).toArray();
    }

    public static int[] argSortDescending(double[] arr){
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        return IntStream.range(0,arr.length)
                .mapToObj(i-> new Pair<>(i,arr[i]))
                .sorted(comparator.reversed())
                .mapToInt(Pair::getFirst).toArray();
    }

    public static int[] argSortDescending(List<Double> arr){
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        return IntStream.range(0,arr.size())
                .mapToObj(i-> new Pair<>(i,arr.get(i)))
                .sorted(comparator.reversed())
                .mapToInt(Pair::getFirst).toArray();
    }



    public static int[] argSortAscending(float[] arr){
        Comparator<Pair<Integer,Float>> comparator = Comparator.comparing(Pair::getSecond);
        return IntStream.range(0,arr.length)
                .mapToObj(i-> new Pair<>(i,arr[i]))
                .sorted(comparator)
                .mapToInt(Pair::getFirst).toArray();
    }

    public static int[] argSortDescending(float[] arr){
        Comparator<Pair<Integer,Float>> comparator = Comparator.comparing(Pair::getSecond);
        return IntStream.range(0,arr.length)
                .mapToObj(i-> new Pair<>(i,arr[i]))
                .sorted(comparator.reversed())
                .mapToInt(Pair::getFirst).toArray();
    }


}
