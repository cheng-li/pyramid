package edu.neu.ccs.pyramid.regression.regression_tree;

import java.util.HashMap;
import java.util.Map;

/**
 * for one leaf node, record lower bounds and upper bounds of different features
 */
public class Bounds {
    private Map<Integer,Double> lowerBounds;
    private Map<Integer,Double> upperBounds;

    private Bounds() {
        this.lowerBounds = new HashMap<>();
        this.upperBounds = new HashMap<>();
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
}
