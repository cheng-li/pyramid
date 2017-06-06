package edu.neu.ccs.pyramid.core.util;

import java.util.ArrayList;
import java.util.List;

public class NgramUtilTest {
    public static void main(String[] args) {
        test4();
    }

    private static void test1(){
        List<String> list = new ArrayList<>();
        list.add("a");
        System.out.println(NgramUtil.toNgram(list));
    }

    private static void test2(){
        List<String> list = new ArrayList<>();
        list.add("a");
        list.add("b");
        System.out.println(NgramUtil.toNgram(list));
    }

    private static void test3(){
        List<String> list = new ArrayList<>();
        list.add("a");
        list.add("b");
        list.add("c");
        System.out.println(NgramUtil.toNgram(list));
    }

    private static void test4(){
        List<String> list = new ArrayList<>();

        System.out.println(NgramUtil.toNgram(list));
    }
}