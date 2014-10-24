package edu.neu.ccs.pyramid.active_learning;

import edu.neu.ccs.pyramid.util.Pair;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/23/14.
 */
public class BestVsSecond {
    private int bestClass;
    private int secondClass;
    private double bestProb;
    private double secondProb;
    private double difference;

    public BestVsSecond(double[] probs) {
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        List<Pair<Integer,Double>> pairs = IntStream.range(0,probs.length)
                .mapToObj(i -> new Pair<Integer, Double>(i, probs[i]))
                .sorted(comparator.reversed()).collect(Collectors.toList());
        this.bestClass = pairs.get(0).getFirst();
        this.bestProb = pairs.get(0).getSecond();
        this.secondClass = pairs.get(1).getFirst();
        this.secondProb = pairs.get(1).getSecond();
        this.difference = this.bestProb - this.secondProb;
    }

    public int getBestClass() {
        return bestClass;
    }

    public int getSecondClass() {
        return secondClass;
    }

    public double getBestProb() {
        return bestProb;
    }

    public double getSecondProb() {
        return secondProb;
    }

    public double getDifference() {
        return difference;
    }

    @Override
    public String toString() {
        return "BestVsSecond{" +
                "bestClass=" + bestClass +
                ", secondClass=" + secondClass +
                ", bestProb=" + bestProb +
                ", secondProb=" + secondProb +
                ", difference=" + difference +
                '}';
    }
}
