package edu.neu.ccs.pyramid.util;

/**
 * Created by chengli on 9/22/16.
 */
public class PrintUtil {
    public static String toMutipleLines(Object[] arr){
        StringBuilder sb = new StringBuilder();
        for (Object obj: arr){
            sb.append(obj).append("\n");
        }
        return sb.toString();
    }

    public static String toMutipleLines(double[] arr){
        StringBuilder sb = new StringBuilder();
        for (double obj: arr){
            sb.append(obj).append("\n");
        }
        return sb.toString();
    }

    public static String toMutipleLines(int[] arr){
        StringBuilder sb = new StringBuilder();
        for (int obj: arr){
            sb.append(obj).append("\n");
        }
        return sb.toString();
    }

}
