package edu.neu.ccs.pyramid.util;

import static org.junit.Assert.*;

public class SigmoidTest {
    public static void main(String[] args) {
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