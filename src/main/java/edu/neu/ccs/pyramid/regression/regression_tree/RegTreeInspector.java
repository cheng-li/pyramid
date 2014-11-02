package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by chengli on 9/4/14.
 */
public class RegTreeInspector {


    /**
     * pair contains feature name and reduction
     * @param tree
     * @return
     */
    public static Map<Integer, Pair<String,Double>> featureImportance(RegressionTree tree){
        Map<Integer, Pair<String,Double>> map = new HashMap<>();
        List<Node> nodes = tree.traverse();
        nodes.stream().filter(node -> !node.isLeaf())
                .forEach(node -> {
                    int featureIndex = node.getFeatureIndex();
                    String featureName = node.getFeatureName();
                    double reduction = node.getReduction();
                    Pair<String,Double> oldPair = map.getOrDefault(featureIndex, new Pair<>(featureName,0.0));
                    Pair<String, Double> newPair = new Pair<>(featureName,oldPair.getSecond()+reduction);
                    map.put(featureIndex, newPair);
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
