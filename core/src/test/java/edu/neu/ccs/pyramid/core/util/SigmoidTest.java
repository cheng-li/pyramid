package edu.neu.ccs.pyramid.core.util;

public class SigmoidTest {
    public static void main(String[] args) {
        System.out.println(Sigmoid.sigmoid(-5));
        System.out.println(Sigmoid.sigmoid(5));
        System.out.println(Sigmoid.sigmoid(-1));
        System.out.println(Sigmoid.sigmoid(0));
        System.out.println(Sigmoid.sigmoid(1));
        System.out.println(Sigmoid.sigmoid(1000));
        System.out.println(Sigmoid.sigmoid(-1000));


        System.out.println(Sigmoid.logSidmoid(0));
        System.out.println(Sigmoid.logSidmoid(1));
        System.out.println(Sigmoid.logSidmoid(1000));
        System.out.println(Sigmoid.logSidmoid(-1000));
    }

}