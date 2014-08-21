package edu.neu.ccs.pyramid.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class ArgSortTest {
    public static void main(String[] args) {
       test1();
        test2();
        testArgSortAscending();
//        testArgSortDescending();
    }

    public static void test1(){
        float[] arr = {1.3f,4.5f,2.3f,2.4f};
        System.out.println(Arrays.toString(ArgSort.argSortAscending(arr)));
    }

    public static void test2(){
        double[] arr = {1.3,4.5,2.3,2.4};
        System.out.println(Arrays.toString(ArgSort.argSortDescending(arr)));
    }

    public static void testArgSortAscending(){
        List<Double> list = new ArrayList<Double>();
        list.add(1.4);
        list.add(3.0);
        list.add(2.1);
        System.out.println(Arrays.toString(ArgSort.argSortAscending(list)));
    }

    public static void testArgSortDescending(){
        List<Double> list = new ArrayList<Double>();
        list.add(1.4);
        list.add(3.0);
        list.add(2.1);
        System.out.println(Arrays.toString(ArgSort.argSortDescending(list)));
    }

}