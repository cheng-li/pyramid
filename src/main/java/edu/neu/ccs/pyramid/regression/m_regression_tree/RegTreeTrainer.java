package edu.neu.ccs.pyramid.regression.m_regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.*;

/**
 * Created by chengli on 8/11/14.
 */
public class RegTreeTrainer {


    public static RegressionTree fit(RegTreeConfig regTreeConfig,
                                     RegDataSet regDataSet){
        return fit(regTreeConfig,regDataSet,regDataSet.getLabels());
    }

    public static RegressionTree fit(RegTreeConfig regTreeConfig,
                                     DataSet dataSet,
                                     double[] labels){
        LeafOutputCalculator leafOutputCalculator = new AverageOutputCalculator(labels);
        return fit(regTreeConfig,dataSet,labels,leafOutputCalculator);
    }

    public static RegressionTree fit(RegTreeConfig regTreeConfig,
                                     DataSet dataSet,
                                     double[] labels,
                                     LeafOutputCalculator leafOutputCalculator){
        RegressionTree tree = new RegressionTree();
        tree.leaves = new ArrayList<>();
        tree.root = new Node();
        //root gets all active data points
        tree.root.setDataAppearance(regTreeConfig.getActiveDataPoints());
        //parallel
        updateNode(tree.root, regTreeConfig,dataSet,labels);
        tree.leaves.add(tree.root);
        tree.root.setLeaf(true);


        /**
         * grow the tree
         */
        int maxNumLeaves = regTreeConfig.getMaxNumLeaves();
        while (tree.leaves.size()<maxNumLeaves) {
            /**
             *find the splitable node which gives the max reduction once split
             */
            Optional<Node> leafToSplitOptional = findLeafToSplit(tree.leaves);
            if (leafToSplitOptional.isPresent()){
                Node leafToSplit = leafToSplitOptional.get();
                splitNode(leafToSplit,regTreeConfig,dataSet,labels,tree.leaves);
            } else {
                break;
            }
        }

        //parallel
        setLeavesOutputs(tree.leaves,leafOutputCalculator);
        cleanLeaves(tree.leaves);
        normalizeReductions(tree,regTreeConfig);
        return tree;
    }

    public static RegressionTree constantTree(double score){
        RegressionTree tree = new RegressionTree();
        tree.root = new Node();
        tree.root.setValue(score);
        tree.root.setLeaf(true);
        tree.leaves.add(tree.root);
        return tree;
    }

    /**
     * split a splitable node
     * @param leafToSplit
     * @param regTreeConfig
     * @param dataSet
     * @param leaves
     */
    private static void splitNode(Node leafToSplit, RegTreeConfig regTreeConfig,
                                  DataSet dataSet, double[] labels,
                                  List<Node> leaves) {


        /**
         * split this leaf node
         */
        int featureIndex = leafToSplit.getFeatureIndex();
        double threshold = leafToSplit.getThreshold();
        Vector inputVector = dataSet.getFeatureColumn(featureIndex).getVector();
        Vector columnVector;
        if (inputVector.isDense()){
            columnVector = inputVector;
        } else {
            columnVector = new DenseVector(inputVector);
        }
        /**
         * create children
         */
        Node leftChild = new Node();
        Node rightChild = new Node();
        int[] parentDataAppearance = leafToSplit.getDataAppearance();

        //update data appearance in children
        //<= go left, > go right


        int[] leftDataAppearance = Arrays.stream(parentDataAppearance).parallel()
                .filter(i -> columnVector.get(i) <= threshold).toArray();
        int[] rightDataAppearance = Arrays.stream(parentDataAppearance).parallel()
                .filter(i -> columnVector.get(i) > threshold).toArray();

        leftChild.setDataAppearance(leftDataAppearance);
        rightChild.setDataAppearance(rightDataAppearance);

        //the last two leaves need not to be updated completely
        //as we don't need to split them later
        int maxNumLeaves = regTreeConfig.getMaxNumLeaves();
        if (leaves.size()!=maxNumLeaves-1){
            updateNode(leftChild,regTreeConfig,dataSet,labels);
            updateNode(rightChild,regTreeConfig,dataSet,labels);
        }


        /**
         * link left and right child to the parent
         */
        leafToSplit.setLeftChild(leftChild);
        leafToSplit.setRightChild(rightChild);

        /**
         * update leaves, remove the parent, and add children
         */
        leafToSplit.setLeaf(false);
        leafToSplit.clearDataAppearance();
        leaves.remove(leafToSplit);
        leftChild.setLeaf(true);
        rightChild.setLeaf(true);
        leaves.add(leftChild);
        leaves.add(rightChild);
    }

    /**
     * parallel
     * given dataAppearance, fill other information
     * @param node
     */
    private static void updateNode(Node node,
                                   RegTreeConfig regTreeConfig,
                                   DataSet dataSet,
                                   double[] labels) {
        Optional<SplitResult> splitResultOptional = Splitter.split(regTreeConfig,
                dataSet,labels,node.getDataAppearance());
        if (splitResultOptional.isPresent()){
            SplitResult splitResult = splitResultOptional.get();
            node.setFeatureIndex(splitResult.getFeatureIndex());
            node.setFeatureName(dataSet.getFeatureColumn(splitResult.getFeatureIndex())
            .getSetting().getFeatureName());
            node.setThreshold(splitResult.getThreshold());
            node.setReduction(splitResult.getReduction());
            node.setSplitable(true);
        } else{
            node.setSplitable(false);
        }
    }

    private static void cleanLeaves(List<Node> leaves){
        for (Node leaf: leaves){
            leaf.clearDataAppearance();
        }
    }


    /**
     * parallel
     */
    private static void setLeavesOutputs(List<Node> leaves, LeafOutputCalculator calculator){
        leaves.parallelStream()
                .forEach(leaf -> setLeafOutput(leaf, calculator));
    }

    private static void setLeafOutput(Node leaf, LeafOutputCalculator calculator){
        int[] dataAppearance = leaf.getDataAppearance();
        double output = calculator.getLeafOutput(dataAppearance);
        leaf.setValue(output);
    }

    private static Optional<Node> findLeafToSplit(List<Node> leaves){
        return leaves.stream().filter(Node::isSplitable)
                .max(Comparator.comparing(Node::getReduction));
    }

    /**
     * does not affect split
     * just make the numbers smaller,
     * and make trees trained with different number of data comparable
     * @param tree
     * @param regTreeConfig
     */
    private static void normalizeReductions(RegressionTree tree, RegTreeConfig regTreeConfig){
        int numDataPoints = regTreeConfig.getActiveDataPoints().length;
        List<Node> nodes = tree.traverse();
        for (Node node: nodes){
            double oldReduction = node.getReduction();
            node.setReduction(oldReduction/numDataPoints);
        }
    }
}
