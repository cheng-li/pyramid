package edu.neu.ccs.pyramid.util;

import java.util.stream.IntStream;

import static org.junit.Assert.*;

public class BernoulliDistributionTest {
    public static void main(String[] args) {
        test4();
    }

    private static void test1(){
        BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(0);
        System.out.println(bernoulliDistribution.probability(0));
        System.out.println(bernoulliDistribution.probability(1));
        System.out.println(bernoulliDistribution.logProbability(0));
        System.out.println(bernoulliDistribution.logProbability(1));
        IntStream.range(0,10).parallel().forEach(i->System.out.println(bernoulliDistribution.sample()));
    }

    private static void test2(){
        BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(1);
        System.out.println(bernoulliDistribution.probability(0));
        System.out.println(bernoulliDistribution.probability(1));
        System.out.println(bernoulliDistribution.logProbability(0));
        System.out.println(bernoulliDistribution.logProbability(1));
        IntStream.range(0,10).parallel().forEach(i->System.out.println(bernoulliDistribution.sample()));
    }

    private static void test3(){
        BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(0.5);
        System.out.println(bernoulliDistribution.probability(0));
        System.out.println(bernoulliDistribution.probability(1));
        System.out.println(bernoulliDistribution.logProbability(0));
        System.out.println(bernoulliDistribution.logProbability(1));
        IntStream.range(0,10).parallel().forEach(i->System.out.println(bernoulliDistribution.sample()));
    }

    private static void test4(){
        BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(0.9);
        System.out.println(bernoulliDistribution.probability(0));
        System.out.println(bernoulliDistribution.probability(1));
        System.out.println(bernoulliDistribution.logProbability(0));
        System.out.println(bernoulliDistribution.logProbability(1));
        IntStream.range(0,10).parallel().forEach(i->System.out.println(bernoulliDistribution.sample()));
    }

}