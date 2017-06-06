package edu.neu.ccs.pyramid.core.util;

import java.util.ArrayList;
import java.util.List;

public class EmpiricalCDFTest {
    public static void main(String[] args) {
        test5();

    }

    private static void test1(){
        List<Double> list = new ArrayList<>();
        for (int i=0;i<10;i++){
            list.add((double)i);
        }
        EmpiricalCDF cdf = new EmpiricalCDF(list,9);
        System.out.println(cdf);
    }

    private static void test2(){
        List<Double> list = new ArrayList<>();
        for (int i=0;i<4;i++){
            list.add((double) i);
        }
        EmpiricalCDF cdf = new EmpiricalCDF(list,3);
        System.out.println(cdf);
    }

    private static void test3(){
        List<Double> list = new ArrayList<>();
        for (int i=0;i<4;i++){
            list.add((double) i);
        }
        for (int i=4;i<10;i++){
            list.add(0.0);
        }
        EmpiricalCDF cdf = new EmpiricalCDF(list,3);
        System.out.println(cdf);
    }

    private static void test4(){
        List<Double> list = new ArrayList<>();
        for (int i=0;i<4;i++){
            list.add((double) i);
        }
        for (int i=4;i<10;i++){
            list.add(0.0);
        }
        EmpiricalCDF cdf = new EmpiricalCDF(list);
        System.out.println(cdf);
    }

    private static void test5(){
        List<Double> list = new ArrayList<>();
        for (int i=0;i<4;i++){
            list.add((double) i);
        }
        for (int i=4;i<10;i++){
            list.add(0.0);
        }
        EmpiricalCDF cdf = new EmpiricalCDF(list,0,7,7);
        System.out.println(cdf);

        List<Double> list2 = new ArrayList<>();
        for (int i=0;i<8;i++){
            list2.add((double) i);
        }
        for (int i=8;i<10;i++){
            list2.add(0.0);
        }
        EmpiricalCDF cdf2 = new EmpiricalCDF(list2,0,7,7);
        System.out.println(cdf2);
        System.out.println(EmpiricalCDF.distance(cdf,cdf2));

    }

}