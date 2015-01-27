package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.active_learning.TrueVsCompetitor;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;

import javax.swing.text.html.Option;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 1/25/15.
 */
public class UncertainSampler {

    private Set<Integer> blackList;

    private List<double[]> classProbMatrix;

    private int numClasses;

    private List<List<Integer>> dataPerClass;

    public UncertainSampler(ClfDataSet clfDataSet) {
        this.blackList = new HashSet<>();
        this.numClasses = clfDataSet.getNumClasses();
        this.dataPerClass = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            dataPerClass.add(new ArrayList<>());
        }

        int[] labels = clfDataSet.getLabels();
        for (int i=0;i<labels.length;i++){
            int label = labels[i];
            dataPerClass.get(label).add(i);
        }

    }

    public void setClassProbMatrix(List<double[]> classProbMatrix) {
        this.classProbMatrix = classProbMatrix;
    }

    public Set<Integer> getBlackList() {
        return blackList;
    }

    public Optional<Integer> getUncertainOne(int classIndex){

        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        List<Integer> sorted = dataPerClass.get(classIndex).stream().map(i -> {
            double difference = new TrueVsCompetitor(classProbMatrix.get(i),classIndex).getDifference();
            return new Pair<>(i,difference);
        }).sorted(comparator).map(Pair::getFirst)
                .collect(Collectors.toList());
        return sorted.stream().filter(i-> !blackList.contains(i)).findFirst();
    }

    public Optional<Integer> getRandomOne(int classIndex){
        List<Integer> dataForClass = new ArrayList<>(dataPerClass.get(classIndex));
        Collections.shuffle(dataForClass);
        return dataForClass.stream().filter(i-> !blackList.contains(i)).findFirst();
    }

    public Optional<Integer> getHardOne(int classIndex){
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        List<Integer> sorted = dataPerClass.get(classIndex).stream().map(i -> new Pair<>(i,classProbMatrix.get(i)[classIndex])
        ).sorted(comparator).map(Pair::getFirst)
                .collect(Collectors.toList());
        return sorted.stream().filter(i-> !blackList.contains(i)).findFirst();
    }

}
