package edu.neu.ccs.pyramid.eval;

import static org.junit.Assert.*;

public class AUCTest {
    public static void main(String[] args) {
        test1();
        test2();
        test3();
    }

    private static void test1(){
        int[] labels = {0,1,1};
        double[] scores = {0.0,0.0,0.0};
        System.out.println(AUC.auc(scores,labels));
    }

    private static void test2(){
        int[] labels = {1,0};
        double[] scores = {0.0,0.0};
        System.out.println(AUC.auc(scores,labels));
    }

    private static void test3(){
        int[] labels = {0,0,1,1};
        double[] scores = {0.0,-10,0.1,0.1};
        System.out.println(AUC.auc(scores,labels));
    }

}