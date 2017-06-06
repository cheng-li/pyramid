package edu.neu.ccs.pyramid.core.regression.regression_tree;

import edu.neu.ccs.pyramid.core.feature.FeatureList;
import edu.neu.ccs.pyramid.core.feature.Feature;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by chengli on 9/4/14.
 */
public class RegTreeInspector {

    /**
     *
     * @param tree
     * @return featureList used in the tree
     */
    public static Set<Integer> features(RegressionTree tree){
        return tree.traverse().stream().filter(node -> !node.isLeaf()).
                map(Node::getFeatureIndex)
                .collect(Collectors.toSet());
    }

    /**
     * pair contains feature name and reduction
     * @param tree
     * @return
     */
//    public static Map<Integer, Pair<String,Double>> featureImportance(RegressionTree tree){
//        List<Feature> featureList = tree.getFeatureList().getAll();
//        Map<Integer, Pair<String,Double>> map = new HashMap<>();
//        List<Node> nodes = tree.traverse();
//        nodes.stream().filter(node -> !node.isLeaf())
//                .forEach(node -> {
//                    int featureIndex = node.getFeatureIndex();
//                    String featureName = featureList.get(node.getFeatureIndex()).getName();
//                    double reduction = node.getReduction();
//                    Pair<String,Double> oldPair = map.getOrDefault(featureIndex, new Pair<>(featureName,0.0));
//                    Pair<String, Double> newPair = new Pair<>(featureName,oldPair.getSecond()+reduction);
//                    map.put(featureIndex, newPair);
//                });
//        return map;
//    }

    public static Map<Feature,Double> featureImportance(RegressionTree tree){
        FeatureList featureList = tree.getFeatureList();
        Map<Feature,Double> map = new HashMap<>();
        List<Node> nodes = tree.traverse();
        nodes.stream().filter(node -> !node.isLeaf())
                .forEach(node -> {
                    int featureIndex = node.getFeatureIndex();
                    Feature feature = featureList.get(featureIndex);
                    double reduction = node.getReduction();
                    double oldValue = map.getOrDefault(feature,0.0);
                    double newValue = reduction+oldValue;
                    map.put(feature,newValue);
                });
        return map;
    }

    /**
     * assume no missing values
     * @param tree
     * @param vector
     * @return
     */
    public static int getMatchedLeaf(RegressionTree tree, Vector vector){
        for (int i=0;i<tree.getNumLeaves();i++){
            double prob = tree.probability(vector,tree.leaves.get(i));
            if (prob==1){
                return i;
            }
        }
        return 0;
    }

    public static List<Integer> getMatchedPath(List<RegressionTree> trees, Vector vector){
        List list = new ArrayList<>();
        for (RegressionTree tree: trees){
            list.add(getMatchedLeaf(tree,vector));
        }
        return list;
    }
}
