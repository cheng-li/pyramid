package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.active_learning.TrueVsCompetitor;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 12/31/14.
 */
public class FocusSetProducer {
    private int numClasses;
    private int numDataPoints;
    private int[][] counts;
    private double[][] gradientMatrix;
    private double[][] classProbMatrix;

    public FocusSetProducer(int numClasses, int numDataPoints) {
        this.numClasses = numClasses;
        this.numDataPoints = numDataPoints;
        this.counts = new int[numClasses][numDataPoints];
    }

    public void setGradientMatrix(double[][] gradientMatrix) {
        this.gradientMatrix = gradientMatrix;
    }

    public void setClassProbMatrix(double[][] classProbMatrix) {
        this.classProbMatrix = classProbMatrix;
    }

    public Set<Integer> produceEasyOnes(int classIndex, int size){
        Set<Integer> set = easyOnes(classIndex,size);
        for (Integer dataIndex: set){
            counts[classIndex][dataIndex] += 1;
        }
        return set;
    }

    public Set<Integer> produceHardOnes(int classIndex, int size){
        Set<Integer> set = hardOnes(classIndex,size);
        for (Integer dataIndex: set){
            counts[classIndex][dataIndex] += 1;
        }
        return set;
    }

    public Set<Integer> produceUncertainOnes(int classIndex, int size){
        Set<Integer> set = uncertainOnes(classIndex,size);
        for (Integer dataIndex: set){
            counts[classIndex][dataIndex] += 1;
        }
        return set;
    }

    private Set<Integer> easyOnes(int classIndex, int size){
        double[] gradientForClass = gradientMatrix[classIndex];
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        List<Integer> sorted = IntStream.range(0, gradientForClass.length)
                .mapToObj(i -> new Pair<>(i, gradientForClass[i]))
                .filter(pair -> pair.getSecond() > 0)
                .sorted(comparator)
                .map(Pair::getFirst)
                .collect(Collectors.toList());
        List<Pair<Integer,Double>> probs = sorted.stream()
                .map(i -> new Pair<>(i, countToProb(counts[classIndex][i])))
                .collect(Collectors.toList());
        return Sampling.rotate(probs,size);
    }

    private Set<Integer> hardOnes(int classIndex, int size){
        double[] gradientForClass = gradientMatrix[classIndex];
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        List<Integer> sorted = IntStream.range(0, gradientForClass.length)
                .mapToObj(i -> new Pair<>(i, gradientForClass[i]))
                .filter(pair -> pair.getSecond() > 0)
                .sorted(comparator.reversed())
                .map(Pair::getFirst)
                .collect(Collectors.toList());
        List<Pair<Integer,Double>> probs = sorted.stream()
                .map(i -> new Pair<>(i, countToProb(counts[classIndex][i])))
                .collect(Collectors.toList());
        return Sampling.rotate(probs,size);
    }


    private Set<Integer> uncertainOnes(int classIndex, int size){
        double[] gradientForClass = gradientMatrix[classIndex];
        List<Integer> matchClass = IntStream.range(0,gradientForClass.length)
                .filter(i-> gradientForClass[i]>0)
                .mapToObj(i->i)
                .collect(Collectors.toList());
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        List<Integer> sorted = matchClass.stream().map(i -> {
            double difference = new TrueVsCompetitor(classProbMatrix[i],classIndex).getDifference();
            return new Pair<>(i,difference);
            }).sorted(comparator).map(Pair::getFirst)
            .collect(Collectors.toList());
        List<Pair<Integer,Double>> probs = sorted.stream()
                .map(i -> new Pair<>(i, countToProb(counts[classIndex][i])))
                .collect(Collectors.toList());
        return Sampling.rotate(probs,size);
    }


    private double countToProb(int count){
        return 1/((double)(count+1));
    }


}
