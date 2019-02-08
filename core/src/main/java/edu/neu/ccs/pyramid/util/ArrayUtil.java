package edu.neu.ccs.pyramid.util;

import java.util.Arrays;
import java.util.List;

public class ArrayUtil {

    public static int[] concatenate(List<int[]> arrays){
        int len = 0;
        for (int[] array: arrays){
            len += array.length;
        }
        int[] con = new int[len];

        int offset = 0;
        for (int[] arr: arrays){
            for (int i=0;i<arr.length;i++){
                con[i+offset] = arr[i];
            }
            offset += arr.length;
        }
        return con;
    }
}
