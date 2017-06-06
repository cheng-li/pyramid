package edu.neu.ccs.pyramid.core.util;

import java.util.HashSet;
import java.util.Set;

public class SetUtilTest {
    public static void main(String[] args) {
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();
        set1.add(1);
        set1.add(2);
        set1.add(3);
        set2.add(2);
        set2.add(3);
        set2.add(4);
        System.out.println(SetUtil.union(set1,set2));
        System.out.println(SetUtil.intersect(set1,set2));
    }

}