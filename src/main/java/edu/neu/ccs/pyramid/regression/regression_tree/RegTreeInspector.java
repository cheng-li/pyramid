package edu.neu.ccs.pyramid.regression.regression_tree;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by chengli on 9/4/14.
 */
public class RegTreeInspector {

    public static Map<Integer, Double> featureImportance(RegressionTree tree){
        Map<Integer, Double> map = new HashMap<>();
        List<Node> nodes = tree.traverse();
        nodes.stream().filter(node -> !node.isLeaf())
                .forEach(node -> {
            int featureIndex = node.getFeatureIndex();
            double reduction = node.getReduction();
            double oldValue = map.getOrDefault(featureIndex, 0.0);
            double newValue = oldValue + reduction;
            map.put(featureIndex, newValue);
        });
        return map;
    }
}
