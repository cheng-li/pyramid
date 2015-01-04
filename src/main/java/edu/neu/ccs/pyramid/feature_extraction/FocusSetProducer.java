package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.active_learning.TrueVsCompetitor;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;

import java.util.*;
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
    private List<Set<Integer>> dataPerClass;
    private boolean promotion = true;

    public FocusSetProducer(int numClasses, int numDataPoints) {
        this.numClasses = numClasses;
        this.numDataPoints = numDataPoints;
        this.counts = new int[numClasses][numDataPoints];
    }

    public void setLabels(int numClasses, int[] labels){
        this.dataPerClass = new ArrayList<>();
        IntStream.range(0,numClasses).forEach(i -> dataPerClass.add(new HashSet<>()));
        IntStream.range(0,labels.length).forEach(i -> {
            int label = labels[i];
            dataPerClass.get(label).add(i);
        });

    }

    public boolean isPromotionEnabled() {
        return promotion;
    }

    public void setPromotion(boolean promotion) {
        this.promotion = promotion;
    }

    public void setGradientMatrix(double[][] gradientMatrix) {
        this.gradientMatrix = gradientMatrix;
    }

    public void setClassProbMatrix(double[][] classProbMatrix) {
        this.classProbMatrix = classProbMatrix;
    }

    public Set<Integer> produceEasyOnes(int classIndex, int size){
        Set<Integer> set = getEasyOnes(classIndex, size);
        if (promotion){
            for (Integer dataIndex: set){
                counts[classIndex][dataIndex] += 1;
            }
        }

        return set;
    }

    public Set<Integer> produceHardOnes(int classIndex, int size){
        Set<Integer> set = getHardOnes(classIndex, size);
        if (promotion){
            for (Integer dataIndex: set){
                counts[classIndex][dataIndex] += 1;
            }
        }
        return set;
    }

    public Set<Integer> produceUncertainOnes(int classIndex, int size){
        Set<Integer> set = getUncertainOnes(classIndex, size);
        if (promotion){
            for (Integer dataIndex: set){
                counts[classIndex][dataIndex] += 1;
            }
        }
        return set;
    }

    public Set<Integer> produceRandomOnes(int classIndex, int size){
        Set<Integer> set = getRandomOnes(classIndex, size);
        if (promotion){
            for (Integer dataIndex: set){
                counts[classIndex][dataIndex] += 1;
            }
        }
        return set;
    }

    public Set<Integer> getEasyOnes(int classIndex, int size){
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

    public Set<Integer> getHardOnes(int classIndex, int size){
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


    public Set<Integer> getUncertainOnes(int classIndex, int size){
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

    public Set<Integer> getRandomOnes(int classIndex, int size){
        List<Integer> matched = dataPerClass.get(classIndex).stream().collect(Collectors.toList());
        Collections.shuffle(matched);
        List<Pair<Integer,Double>> probs = matched.stream()
                .map(i -> new Pair<>(i, countToProb(counts[classIndex][i])))
                .collect(Collectors.toList());
        return Sampling.rotate(probs,size);
    }



    private double countToProb(int count){
        return 1/((double)(count+1));
    }


}
