package edu.neu.ccs.pyramid.util;

/**
 * Created by chengli on 7/10/16.
 */
public class Progress {
    public static String percentage(double current, double total){
        return (int)Math.floor(current*100/total)+"%";
    }
}
