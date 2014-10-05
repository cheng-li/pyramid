package edu.neu.ccs.pyramid.classification.ecoc;

import java.util.Arrays;

import static org.junit.Assert.*;

public class CodeMatrixTest {
    public static void main(String[] args) {
//        test1();
//        test2();
        test3();
    }

    static void test1(){
        CodeMatrix codeMatrix = CodeMatrix.exhaustiveCodes(5);
        System.out.println(codeMatrix);
    }

    static void test2(){
        CodeMatrix codeMatrix = CodeMatrix.exhaustiveCodes(5);
        System.out.println(codeMatrix);
        int[] oldLabels = {0,1,2,3,4,2,1,0,3};
        int[] newLabels = codeMatrix.aggregateLabels(2,oldLabels);
        System.out.println(Arrays.toString(oldLabels));
        System.out.println(Arrays.toString(newLabels));
    }

    static void test3(){
        CodeMatrix codeMatrix = CodeMatrix.exhaustiveCodes(5);
        System.out.println(codeMatrix);
        int[] code = {0,1,1,0,1,1,0,1,0,1,1,0,0,1,0};
        int matched = codeMatrix.matchClass(code);
        System.out.println(matched);
    }

}