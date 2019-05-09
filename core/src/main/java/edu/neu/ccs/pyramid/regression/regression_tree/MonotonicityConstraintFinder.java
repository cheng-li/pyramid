package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.SetUtil;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class MonotonicityConstraintFinder {

    /**
     * deal with numerical precision issue
     * @param a
     * @param b
     * @return
     */
    static boolean leqWithSlack(double a, double b){
        return (b-a)>=-0.00001;
    }

    public static List<Pair<Integer,Integer>> findViolatingPairs(List<Node> leaves, int[] monotonicity){
        List<Pair<Integer,Integer>> comparablePairs = findComparablePairs(leaves, monotonicity);
        List<Pair<Integer,Integer>> violatingPairs = new ArrayList<>();
        for (Pair<Integer,Integer> pair: comparablePairs){
            Node node1 = leaves.get(pair.getFirst());
            Node node2 = leaves.get(pair.getSecond());
            if (!(leqWithSlack(node1.getValue(),node2.getValue()))){
                violatingPairs.add(pair);
            }
        }
        return violatingPairs;
    }



    public static boolean isMonotonic(List<Node> leaves, int[] monotonicity){
        List<Pair<Integer,Integer>> pairs = findComparablePairs(leaves, monotonicity);

        for (Pair<Integer,Integer> pair: pairs){
            Node node1 = leaves.get(pair.getFirst());
            Node node2 = leaves.get(pair.getSecond());
            if (!(leqWithSlack(node1.getValue(),node2.getValue()))){
                return false;
            }
        }
        return true;
    }

    public static boolean isMonotonic(RegressionTree tree, int[] monotonicity){
        List<Node> leaves = tree.getLeaves();
        return isMonotonic(leaves, monotonicity);
    }

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
