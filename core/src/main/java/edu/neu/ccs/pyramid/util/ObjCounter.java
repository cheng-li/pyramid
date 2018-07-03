package edu.neu.ccs.pyramid.util;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class ObjCounter<E> {
    private HashMap<E,Integer> counts;

    public ObjCounter() {
        this.counts = new HashMap<>();
    }

    synchronized public void add (E object){
        int oldCount = counts.getOrDefault(object,0);
        counts.put(object,oldCount+1);
    }

    synchronized public void add (E object, int amount){
        int oldCount = counts.getOrDefault(object,0);
        counts.put(object,oldCount+amount);
    }

    public List<Pair<E,Integer>> iterateByCountIncreasing(){
        Comparator<Pair<E,Integer>> comparator = Comparator.comparing(Pair::getSecond);
        return counts.entrySet().stream().map(entry->new Pair<>(entry.getKey(),entry.getValue()))
                .sorted(comparator).collect(Collectors.toList());
    }


    public List<Pair<E,Integer>> iterateByCountDecreasing(){
        Comparator<Pair<E,Integer>> comparator = Comparator.comparing(Pair::getSecond);
        return counts.entrySet().stream().map(entry->new Pair<>(entry.getKey(),entry.getValue()))
                .sorted(comparator.reversed()).collect(Collectors.toList());
    }

    public List<Pair<E,Double>> iterateByPercentageIncreasing(){
        double total = counts.entrySet().stream().mapToDouble(Map.Entry::getValue).sum();
        Comparator<Pair<E,Double>> comparator = Comparator.comparing(Pair::getSecond);
        return counts.entrySet().stream().map(entry->new Pair<>(entry.getKey(),entry.getValue()/total))
                .sorted(comparator).collect(Collectors.toList());
    }


    public List<Pair<E,Double>> iterateByPercentageDecreasing(){
        double total = counts.entrySet().stream().mapToDouble(Map.Entry::getValue).sum();
        Comparator<Pair<E,Double>> comparator = Comparator.comparing(Pair::getSecond);
        return counts.entrySet().stream().map(entry->new Pair<>(entry.getKey(),entry.getValue()/total))
                .sorted(comparator.reversed()).collect(Collectors.toList());
    }
}
