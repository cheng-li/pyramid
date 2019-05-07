package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.feature.FeatureList;

import java.util.*;
import java.util.stream.Collectors;

/**
 * for one leaf node, record lower bounds and upper bounds of different features
 */
public class Bounds {
    private Map<Integer,Double> lowerBounds;
    private Map<Integer,Double> upperBounds;
    Set<Integer> usedFeatures;

    private Bounds() {
        this.lowerBounds = new HashMap<>();
        this.upperBounds = new HashMap<>();
        this.usedFeatures = new HashSet<>();
    }


    public Bounds(Node leaf){
        this();
        Node node = leaf;
        while(true){
            if (node.getParent()==null){
                break;
            }
            Node parent = node.getParent();
            int featureIndex = parent.getFeatureIndex();
            usedFeatures.add(featureIndex);
            double threshold = parent.getThreshold();
            if (node==parent.getLeftChild()){
                double existingUpper = upperBounds.getOrDefault(featureIndex,Double.POSITIVE_INFINITY);
                upperBounds.put(featureIndex,Math.min(existingUpper,threshold));
            }
            if (node==parent.getRightChild()){
                double existingLower = lowerBounds.getOrDefault(featureIndex,Double.NEGATIVE_INFINITY);
                lowerBounds.put(featureIndex,Math.max(existingLower,threshold));
            }
            node = parent;
        }
    }

    void setLowerBound(int featureIndex, double lowerBound){
        lowerBounds.put(featureIndex,lowerBound);
    }

    void setUpperBound(int featureIndex, double upperBound){
        upperBounds.put(featureIndex,upperBound);
    }

    double getLowerBound(int featureIndex){
        return lowerBounds.getOrDefault(featureIndex,Double.NEGATIVE_INFINITY);
    }

    double getUpperBound(int featureIndex){
        return upperBounds.getOrDefault(featureIndex,Double.POSITIVE_INFINITY);
    }

    Set<Integer> getUsedFeatures() {
        return usedFeatures;
    }


    @Override
    public String toString() {
        List<Integer> usedFeaturesSorted = usedFeatures.stream().sorted().collect(Collectors.toList());
        final StringBuilder sb = new StringBuilder();
        for (int j: usedFeaturesSorted){
            sb.append("feature "+j).append(":").append("(")
                    .append(getLowerBound(j)).append(",").append(getUpperBound(j))
                    .append("]").append("  ");
        }
        return sb.toString();
    }

    public String toString(FeatureList featureList) {
        List<Integer> usedFeaturesSorted = usedFeatures.stream().sorted().collect(Collectors.toList());
        final StringBuilder sb = new StringBuilder();
        for (int j: usedFeaturesSorted){
            sb.append("feature "+featureList.get(j).getName()).append(":").append("(")
                    .append(getLowerBound(j)).append(",").append(getUpperBound(j))
                    .append("]").append("  ");
        }
        return sb.toString();
    }
}
