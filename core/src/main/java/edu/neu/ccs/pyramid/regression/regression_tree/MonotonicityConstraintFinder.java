package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.SetUtil;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class MonotonicityConstraintFinder {

    public static List<Pair<Integer,Integer>> findComparablePairs(List<Node> leaves, int[] monotonicity){
        List<Bounds> boundsList = leaves.stream().map(Bounds::new).collect(Collectors.toList());
        List<Pair<Integer,Integer>> pairs = new ArrayList<>();
        for (int i=0;i<leaves.size();i++){
            for (int j=0;j<leaves.size();j++){
                if (i!=j){
                    Bounds bounds1 = boundsList.get(i);
                    Bounds bounds2 = boundsList.get(j);
                    if (leq(bounds1, bounds2, monotonicity)){
                        pairs.add(new Pair<>(i,j));
                    }
                }
            }
        }
        return pairs;
    }

    private static boolean leq(Bounds smallOne, Bounds bigOne, int[] monotonicity){
        Set<Integer> usedFeatures = new HashSet<>();
        usedFeatures.addAll(smallOne.getUsedFeatures());
        usedFeatures.addAll(bigOne.getUsedFeatures());
        for (int j: usedFeatures){
            if (monotonicity[j]==1){
                //monotonic feature
                if (! (smallOne.getLowerBound(j)<bigOne.getUpperBound(j))){
                    return false;
                }
            }

            if (monotonicity[j]==0){
                //non monotonic feature
                if (!((smallOne.getLowerBound(j)<bigOne.getUpperBound(j)) && (bigOne.getLowerBound(j)<smallOne.getUpperBound(j)))){
                    return false;
                }
            }
        }

        return true;

    }
}
