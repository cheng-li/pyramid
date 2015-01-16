package edu.neu.ccs.pyramid.feature_extraction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class NgramEnumeratorTest {
    public static void main(String[] args) {
//        test1();
//        test2();
        test3();
    }

    private static void test1(){
        List<String> list = new ArrayList<>();
        list.add("a");
        list.add("b");
        list.add("c");
        list.add("d");
        list.add("e");
        System.out.println(NgramEnumerator.toNgram(list,1,3));
    }

    private static void test2(){
        List<String> list = new ArrayList<>();
        Map<String,Integer> counts = new HashMap<>();
        list.add("a");
        list.add("b");
        list.add("a");
        list.add("b");
        list.add("c");
        list.add("d");
        list.add("e");
        NgramEnumerator.updateNgramCounts(list, 2, counts);
        System.out.println(counts);
    }

    private static void test3(){
        Map<Integer, String> tv = new HashMap<>();
        tv.put(0,"a");
        tv.put(1,"b");
//        tv.put(2,"c");
        tv.put(3,"d");
        tv.put(4,"e");
        tv.put(5,"f");
        tv.put(6,"g");
        tv.put(7,"a");
        tv.put(8,"b");
        System.out.println(NgramEnumerator.getNgramCounts(tv,2));
    }

}