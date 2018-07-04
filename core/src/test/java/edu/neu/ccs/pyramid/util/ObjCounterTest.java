package edu.neu.ccs.pyramid.util;


public class ObjCounterTest  {
    public static void main(String[] args) {
        ObjCounter<String> objCounter = new ObjCounter<>();
        objCounter.add("a",3);
        objCounter.add("b");
        objCounter.add("c",2);
        System.out.println(objCounter.iterateByCountIncreasing());
        System.out.println(objCounter.iterateByPercentageDecreasing());
    }

}