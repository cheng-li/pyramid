package edu.neu.ccs.pyramid.util;

import junit.framework.TestCase;

import java.util.Arrays;

/**
 * Created by chengli on 10/2/16.
 */
public class MathUtilTest{
    public static void main(String[] args) {

        test6();

    }

    private static void test1(){
        double[] p  = {0.5, 0.5};
        System.out.println(Arrays.toString(MathUtil.inverseSoftMax(p)));
    }

    private static void test2(){
        double[] p  = {0.3, 0.7};
        System.out.println(Arrays.toString(MathUtil.inverseSoftMax(p)));
    }

    private static void test3(){
        double[] p  = {0, 1};
        System.out.println(Arrays.toString(MathUtil.inverseSoftMax(p)));
    }

    private static void test4(){
        double[] s = {-5,5};
        System.out.println(Arrays.toString(MathUtil.softmax(s)));
    }

    private static void test5(){

        System.out.println(MathUtil.inverseSigmoid(0));
        System.out.println(MathUtil.inverseSigmoid(1));
        System.out.println(MathUtil.inverseSigmoid(0.5));
        System.out.println(MathUtil.inverseSigmoid(0.1));
        System.out.println(MathUtil.inverseSigmoid(0.9));
        System.out.println(MathUtil.inverseSigmoid(1.0/1000000));

        double[] s = {1,2,3,4,5,10,100};
        System.out.println(MathUtil.median(s));
    }

    private static void test6(){
        double[] s = {1,2,3,4,5};
        double[] w = {0.1, 0.5, 0.1, 0.1, 0.2};
        System.out.println(MathUtil.weightedMedian(s, w));
    }


}