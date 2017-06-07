package edu.neu.ccs.pyramid.util;

import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 10/28/16.
 */
public class PrintUtilTest{
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        List<Double> list = new ArrayList<>();
        for (int i=0;i<10;i++){
            list.add(Math.exp(i));

        }

        System.out.println(PrintUtil.printWithIndex(list,1));
    }

}