package edu.neu.ccs.pyramid.feature_extraction;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.feature.Ngram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

public class NgramEnumeratorTest {
    public static void main(String[] args) {
//        test1();
//        test2();
//        test3();
//    test4();
//    test7();
    test8();
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

//    private static void test4(){
//        System.out.println(NgramEnumerator.createTemplate(3, 1));
//        System.out.println(NgramEnumerator.createTemplate(1, 0));
//        System.out.println(NgramEnumerator.createTemplate(1, 1));
//        System.out.println(NgramEnumerator.createTemplate(2, 0));
//        System.out.println(NgramEnumerator.createTemplate(2, 1));
//        System.out.println(NgramEnumerator.createTemplate(2, 2));
//        System.out.println(NgramEnumerator.createTemplate(3, 2));
//    }
//
//    private static void test5(){
//        List<List<Integer>> templates = NgramEnumerator.createTemplate(3, 1);
//        System.out.println("templates = "+ templates);
//        List<String> source = new ArrayList<>();
//        for (int i=0;i<10;i++){
//            source.add(""+i);
//        }
//        for (List<Integer> template: templates){
//            System.out.println("template = "+template);
//            Multiset<Ngram> multiset = ConcurrentHashMultiset.create();
//            NgramEnumerator.add(source,multiset,"body",1,template);
//            System.out.println(multiset);
//        }
//
//
//
//    }
//
//    private static void test6(){
//        List<List<Integer>> templates = NgramEnumerator.createTemplate(3, 2);
//        System.out.println("templates = "+ templates);
//        List<String> source = new ArrayList<>();
//        for (int i=0;i<10;i++){
//            source.add(""+i);
//        }
//        for (List<Integer> template: templates){
//            System.out.println("template = "+template);
//            Multiset<Ngram> multiset = ConcurrentHashMultiset.create();
//            NgramEnumerator.add(source,multiset,"body",1,template);
//            System.out.println(multiset);
//        }
//
//
//
//    }
//
//    private static void test7(){
//        List<List<Integer>> templates = NgramEnumerator.createTemplate(1, 0);
//        System.out.println("templates = "+ templates);
//        List<String> source = new ArrayList<>();
//        for (int i=0;i<10;i++){
//            source.add(""+i);
//        }
//        for (List<Integer> template: templates){
//            System.out.println("template = "+template);
//            Multiset<Ngram> multiset = ConcurrentHashMultiset.create();
//            NgramEnumerator.add(source,multiset,"body",1,template);
//            System.out.println(multiset);
//        }
//    }

    private static void test8(){
        NgramTemplate template = new NgramTemplate("body",3,1);
        Multiset<Ngram> multiset = ConcurrentHashMultiset.create();
        List<String> source = new ArrayList<>();
        for (int i=0;i<10;i++){
            source.add(""+i);
        }
        NgramEnumerator.add(source,multiset,template);
        System.out.println(multiset.elementSet().stream().map(Ngram::getNgram).collect(Collectors.toList()));
    }

}