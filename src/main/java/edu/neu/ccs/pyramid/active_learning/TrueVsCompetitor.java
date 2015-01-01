package edu.neu.ccs.pyramid.active_learning;

import edu.neu.ccs.pyramid.util.Pair;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 12/31/14.
 */
public class TrueVsCompetitor {
    private int bestClass;
    private int secondClass;
    private double bestProb;
    private double secondProb;
    private double difference;
    private double trueProb;
    private double competitorProb;

    public TrueVsCompetitor(double[] probs, int trueLabel) {
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        List<Pair<Integer,Double>> pairs = IntStream.range(0, probs.length)
                .mapToObj(i -> new Pair<Integer, Double>(i, probs[i]))
                .sorted(comparator.reversed()).collect(Collectors.toList());
        this.bestClass = pairs.get(0).getFirst();
        this.bestProb = pairs.get(0).getSecond();
        this.secondClass = pairs.get(1).getFirst();
        this.secondProb = pairs.get(1).getSecond();
        this.trueProb = probs[trueLabel];
        if (trueLabel==bestClass){
            this.competitorProb = secondProb;
        } else {
            competitorProb = bestProb;
        }
        difference = Math.abs(trueProb - competitorProb);
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
        return "TrueVsCompetitor{" +
                "bestClass=" + bestClass +
                ", secondClass=" + secondClass +
                ", bestProb=" + bestProb +
                ", secondProb=" + secondProb +
                ", difference=" + difference +
                ", trueProb=" + trueProb +
                ", competitorProb=" + competitorProb +
                '}';
    }
}
