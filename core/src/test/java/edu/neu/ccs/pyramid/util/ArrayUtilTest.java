package edu.neu.ccs.pyramid.util;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ArrayUtilTest {
    public static void main(String[] args) {
        int[] a = {1,2};
        int[] b = {3,4,5};
        int[] c = {6,7,8,0};
        List<int[]> arrs = new ArrayList<>();
        arrs.add(a);
        arrs.add(b);
        arrs.add(c);
        System.out.println(Arrays.toString(ArrayUtil.concatenate(arrs)));
    }

}