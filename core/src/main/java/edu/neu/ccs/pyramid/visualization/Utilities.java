package edu.neu.ccs.pyramid.visualization;


import java.util.Map;

/**
 * Created by shikhar on 6/28/17.
 */
public class Utilities {
    public static void echo (Object o){
        System.out.println("  >> " + o);
    }
    public static void error (Object o) { System.err.println(">> Error : "+ o);}

    public static <K, V> void displayMap(Map<K, V> map) {
        map.entrySet()
                .stream()
                .forEach(entrySet -> System.out.println(entrySet.getKey()+" : "+ entrySet.getValue()));
    }
}