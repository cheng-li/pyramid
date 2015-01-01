package edu.neu.ccs.pyramid.util;

import java.util.ArrayList;
import java.util.List;

public class SamplingTest {
    public static void main(String[] args) {
        test4();
    }

    private static void test1(){
        System.out.println(Sampling.doubleUniform(0, 10));
    }

    private static void test2(){
        System.out.println(Sampling.doubleLogUniform(0.001, 1));
    }

    private static void test3(){
        System.out.println(Sampling.intUniform(3,4));
    }

    private static void test4(){
        List<Pair<Integer, Double>> probs = new ArrayList<>();
        probs.add(new Pair<>(0,1.0));
        probs.add(new Pair<>(1,0.1));
        probs.add(new Pair<>(2,0.2));
        System.out.println(Sampling.rotate(probs,2));
    }

}