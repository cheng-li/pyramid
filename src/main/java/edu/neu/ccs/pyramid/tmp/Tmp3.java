package edu.neu.ccs.pyramid.tmp;

import edu.neu.ccs.pyramid.util.Pair;

import java.util.stream.IntStream;

/**
 * Created by chengli on 9/8/14.
 */
public class Tmp3 {
    public static void main(String[] args) {
        System.out.println(IntStream.range(0, 100).parallel().map(i -> i)
                .average().getAsDouble());
    }
}
