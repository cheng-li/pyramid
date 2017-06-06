package edu.neu.ccs.pyramid.core.util;

public class GridTest {
    public static void main(String[] args) {
//        test1();
//        test2();
        test3();
    }

    private static void test1(){
        System.out.println(Grid.uniform(0,10,11));
        System.out.println(Grid.uniform(0,1,5));
    }

    private static void test2(){
        System.out.println(Grid.logUniform(0.1,10,3));
        System.out.println(Grid.logUniform(0.1,1000,5));
    }


    private static void test3(){
        System.out.println(Grid.uniformDecreasing(0,10,11));

    }
}