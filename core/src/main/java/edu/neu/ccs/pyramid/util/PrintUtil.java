package edu.neu.ccs.pyramid.util;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Created by chengli on 9/22/16.
 */
public class PrintUtil {

    public static String format(double d){
        if (d==0){
            return "0";
        }

        if (d>0.01){
        DecimalFormat df = new DecimalFormat("0.###");
        return df.format(d);
        } else {
            DecimalFormat df = new DecimalFormat("#.##E0");
            return df.format(d);
        }

    }


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

    public static String toSimpleString(double[] arr){
        return Arrays.toString(arr).replaceAll(Pattern.quote("["),"").replaceAll(Pattern.quote("]"),"");
    }

    public static String printWithIndex(List<? extends Object> list, int startIndex){
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i=0;i<list.size();i++){
            sb.append(""+(i+startIndex)).append(":").append(list.get(i));
            if (i!=list.size()-1){
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    public static String printWithIndex(List<? extends Object> list){
        return printWithIndex(list, 0);
    }


    public static String printWithIndex(double[] arr){
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i=0;i<arr.length;i++){
            sb.append(i).append(":").append(arr[i]);
            if (i!=arr.length-1){
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }


    public static String printWithIndex(int[] arr){
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i=0;i<arr.length;i++){
            sb.append(i).append(":").append(arr[i]);
            if (i!=arr.length-1){
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
