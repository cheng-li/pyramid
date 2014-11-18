package edu.neu.ccs.pyramid.util;

public class SamplingTest {
    public static void main(String[] args) {
        test3();
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

}