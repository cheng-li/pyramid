package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.*;

import java.util.stream.IntStream;
import java.util.stream.Stream;

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
        LeafOutputCalculator leafOutputCalculator = new AverageOutputCalculator();
        return fit(regTreeConfig,dataSet,labels,leafOutputCalculator);
    }

    public static RegressionTree fit(RegTreeConfig regTreeConfig,
                                     DataSet dataSet,
                                     double[] labels,
                                     LeafOutputCalculator leafOutputCalculator){
        double[] weights = new double[labels.length];
        Arrays.fill(weights,1.0);
        return fit(regTreeConfig,dataSet,labels,weights,leafOutputCalculator);
    }


    public static RegressionTree fit(RegTreeConfig regTreeConfig,
                                     DataSet dataSet,
                                     double[] labels,
                                     double[] weights,
                                     LeafOutputCalculator leafOutputCalculator){
        RegressionTree tree = new RegressionTree();
        tree.setFeatureList(dataSet.getFeatureList());

        tree.leaves = new ArrayList<>();
        tree.root = new Node();
        tree.root.setId(tree.numNodes);
        tree.numNodes += 1;


        //root gets all active data points
        double[] rootProbs = new double[dataSet.getNumDataPoints()];
        for (int dataPoint=0; dataPoint<dataSet.getNumDataPoints();dataPoint++){
            rootProbs[dataPoint]=weights[dataPoint];
        }
        tree.root.setProbs(rootProbs);
        //parallel
        updateNode(tree.root, regTreeConfig,dataSet,labels);
        tree.leaves.add(tree.root);
        tree.root.setLeaf(true);
        tree.allNodes.add(tree.root);

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
                splitNode(tree, leafToSplit,regTreeConfig,dataSet,labels);
            } else {
                break;
            }
        }

        //parallel
        setLeavesOutputs(regTreeConfig, tree.leaves,leafOutputCalculator, labels);
        cleanLeaves(tree.leaves);
        normalizeReductions(tree,dataSet);
        return tree;
    }


    public static RegressionTree fit(RegTreeConfig regTreeConfig,
                                     DataSet dataSet,
                                     double[] labels,
                                     double[] weights,
                                     LeafOutputCalculator leafOutputCalculator,
                                     int[] monotonicity){
        RegressionTree tree = new RegressionTree();
        tree.setFeatureList(dataSet.getFeatureList());

        tree.leaves = new ArrayList<>();
        tree.root = new Node();
        tree.root.setId(tree.numNodes);
        tree.numNodes += 1;


        //root gets all active data points
        double[] rootProbs = new double[dataSet.getNumDataPoints()];
        for (int dataPoint=0; dataPoint<dataSet.getNumDataPoints();dataPoint++){
            rootProbs[dataPoint]=weights[dataPoint];
        }
        tree.root.setProbs(rootProbs);
        //parallel
        if (regTreeConfig.getMonotonicityType().equals("xgboost")){
            updateNode(tree.root, regTreeConfig,dataSet,labels, monotonicity);
        } else {
            updateNode(tree.root, regTreeConfig,dataSet,labels);
        }

        leafOutputCalculator.setParallel(regTreeConfig.isParallel());
        setLeafOutput(tree.root,leafOutputCalculator,labels);

        tree.leaves.add(tree.root);
        tree.root.setLeaf(true);
        tree.allNodes.add(tree.root);

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
                if (regTreeConfig.getMonotonicityType().equals("xgboost")){
                    splitNode(tree, leafToSplit,regTreeConfig,dataSet,labels, monotonicity, leafOutputCalculator);
                } else {
                    splitNode(tree, leafToSplit,regTreeConfig,dataSet,labels);
                }

            } else {
                break;
            }
        }

        setLeavesOutputs(regTreeConfig, tree.leaves,leafOutputCalculator, labels);

        if (regTreeConfig.getMonotonicityType().equals("weak")){
            MonotonicityPostProcessor.changeOutput(tree.leaves,monotonicity,false);
        }

        if (regTreeConfig.getMonotonicityType().equals("strong")){
            MonotonicityPostProcessor.changeOutput(tree.leaves,monotonicity,true);
        }


        cleanLeaves(tree.leaves);
        normalizeReductions(tree,dataSet);
        return tree;
    }

    public static RegressionTree constantTree(double score){
        RegressionTree tree = new RegressionTree();
        tree.root = new Node();
        tree.root.setValue(score);
        tree.root.setLeaf(true);
        tree.leaves.add(tree.root);
        tree.allNodes.add(tree.root);
        return tree;
    }

    /**
     * split a splitable node
     * @param leafToSplit
     * @param regTreeConfig
     * @param dataSet
     */
    private static void splitNode(RegressionTree tree, Node leafToSplit, RegTreeConfig regTreeConfig,
                                  DataSet dataSet, double[] labels) {
        int numDataPoints = dataSet.getNumDataPoints();

        /**
         * split this leaf node
         */
        int featureIndex = leafToSplit.getFeatureIndex();
        double threshold = leafToSplit.getThreshold();
        Vector inputVector = dataSet.getColumn(featureIndex);
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
        leftChild.setId(tree.numNodes);
        tree.numNodes += 1;
        Node rightChild = new Node();
        rightChild.setId(tree.numNodes);
        tree.numNodes += 1;

        double[] parentProbs = leafToSplit.getProbs();
        double[] leftProbs = new double[numDataPoints];
        double[] rightProbs = new double[numDataPoints];
        IntStream intStream = IntStream.range(0,numDataPoints);
        if (regTreeConfig.isParallel()){
            intStream = intStream.parallel();
        }
        intStream.forEach(i->{
            double featureValue = columnVector.get(i);
            if (Double.isNaN(featureValue)){
                // go to both branches probabilistically
                leftProbs[i] = parentProbs[i]*leafToSplit.getLeftProb();
                rightProbs[i] = parentProbs[i]*leafToSplit.getRightProb();
            } else {
                //<= go left, > go right
                if (featureValue<=threshold){
                    leftProbs[i] = parentProbs[i];
                    rightProbs[i] = 0;
                } else {
                    leftProbs[i] = 0;
                    rightProbs[i] = parentProbs[i];
                }
            }
        });

        leftChild.setProbs(leftProbs);
        rightChild.setProbs(rightProbs);


        //the last two leaves need not to be updated completely
        //as we don't need to split them later
        int maxNumLeaves = regTreeConfig.getMaxNumLeaves();
        if (tree.leaves.size()!=maxNumLeaves-1){
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
        leafToSplit.clearProbs();
        tree.leaves.remove(leafToSplit);
        leftChild.setLeaf(true);
        rightChild.setLeaf(true);
        tree.leaves.add(leftChild);
        tree.leaves.add(rightChild);
        tree.allNodes.add(leftChild);
        tree.allNodes.add(rightChild);
    }

    /**
     * xgboost monotonicity
     * split a splitable node
     * @param leafToSplit
     * @param regTreeConfig
     * @param dataSet
     */
    private static void splitNode(RegressionTree tree, Node leafToSplit, RegTreeConfig regTreeConfig,
                                  DataSet dataSet, double[] labels, int[] monotonicity, LeafOutputCalculator leafOutputCalculator) {
        int numDataPoints = dataSet.getNumDataPoints();

        /**
         * split this leaf node
         */
        int featureIndex = leafToSplit.getFeatureIndex();
        double threshold = leafToSplit.getThreshold();
        Vector inputVector = dataSet.getColumn(featureIndex);
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
        leftChild.setId(tree.numNodes);
        tree.numNodes += 1;
        Node rightChild = new Node();
        rightChild.setId(tree.numNodes);
        tree.numNodes += 1;

        double[] parentProbs = leafToSplit.getProbs();
        double[] leftProbs = new double[numDataPoints];
        double[] rightProbs = new double[numDataPoints];
        IntStream intStream = IntStream.range(0,numDataPoints);
        if (regTreeConfig.isParallel()){
            intStream = intStream.parallel();
        }
        intStream.forEach(i->{
            double featureValue = columnVector.get(i);
            if (Double.isNaN(featureValue)){
                // go to both branches probabilistically
                leftProbs[i] = parentProbs[i]*leafToSplit.getLeftProb();
                rightProbs[i] = parentProbs[i]*leafToSplit.getRightProb();
            } else {
                //<= go left, > go right
                if (featureValue<=threshold){
                    leftProbs[i] = parentProbs[i];
                    rightProbs[i] = 0;
                } else {
                    leftProbs[i] = 0;
                    rightProbs[i] = parentProbs[i];
                }
            }
        });

        leftChild.setProbs(leftProbs);
        rightChild.setProbs(rightProbs);


        //the last two leaves need not to be updated completely
        //as we don't need to split them later
        int maxNumLeaves = regTreeConfig.getMaxNumLeaves();
        if (tree.leaves.size()!=maxNumLeaves-1){
            updateNode(leftChild,regTreeConfig,dataSet,labels, monotonicity);
            updateNode(rightChild,regTreeConfig,dataSet,labels, monotonicity);
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
        leafToSplit.clearProbs();
        tree.leaves.remove(leafToSplit);
        leftChild.setLeaf(true);
        rightChild.setLeaf(true);
        tree.leaves.add(leftChild);
        tree.leaves.add(rightChild);
        tree.allNodes.add(leftChild);
        tree.allNodes.add(rightChild);

        int mono = monotonicity[featureIndex];
        leafOutputCalculator.setParallel(regTreeConfig.isParallel());
        setLeafOutput(leftChild,leafOutputCalculator,labels);
        setLeafOutput(rightChild,leafOutputCalculator,labels);

        setBoundForChildren(leafToSplit,mono);

    }

    // xgboost monotonicity
    private static void setBoundForChildren(Node nodeToSplit, int monotonicity){
        Node leftChild = nodeToSplit.getLeftChild();
        Node rightChild = nodeToSplit.getRightChild();
        double lowerBound = nodeToSplit.getLowerBound();
        double upperBound = nodeToSplit.getUpperBound();

        // first inherit bounds
        leftChild.setLowerBound(lowerBound);
        leftChild.setUpperBound(upperBound);

        rightChild.setLowerBound(lowerBound);
        rightChild.setUpperBound(upperBound);

        // correct output if out of bound
        leftChild.boundValue();
        rightChild.boundValue();

        // tighten bounds

        //do nothing if monotonicity=0
//        if (monotonicity==0){
//            leftChild.setLowerBound(lowerBound);
//            leftChild.setUpperBound(upperBound);
//            rightChild.setLowerBound(lowerBound);
//            rightChild.setUpperBound(upperBound);
//        }


        if (monotonicity==1){
            double mid = (leftChild.getValue()+rightChild.getValue())*0.5;
//            leftChild.setLowerBound(lowerBound);
            leftChild.setUpperBound(Math.min(upperBound,mid));
            rightChild.setLowerBound(Math.max(lowerBound,mid));
//            rightChild.setUpperBound(upperBound);
        }

        if (monotonicity==-1){
            double mid = (leftChild.getValue()+rightChild.getValue())*0.5;
            leftChild.setLowerBound(Math.max(lowerBound,mid));
//            leftChild.setUpperBound(upperBound);
//            rightChild.setLowerBound(lowerBound);
            rightChild.setUpperBound(Math.min(upperBound,mid));
        }
    }



    /**
     * parallel
     * given probs, fill other information
     * @param node
     */
    private static void updateNode(Node node,
                                   RegTreeConfig regTreeConfig,
                                   DataSet dataSet,
                                   double[] labels) {
        Optional<SplitResult> splitResultOptional = Splitter.split(regTreeConfig,
                dataSet,labels,node.getProbs());
        if (splitResultOptional.isPresent()){
            SplitResult splitResult = splitResultOptional.get();
            node.setFeatureIndex(splitResult.getFeatureIndex());
            node.setThreshold(splitResult.getThreshold());
            node.setReduction(splitResult.getReduction());
            double leftCount = splitResult.getLeftCount();
            double rightCount = splitResult.getRightCount();
            double totalCount = leftCount + rightCount;
            node.setLeftProb(leftCount/totalCount);
            node.setRightProb(rightCount/totalCount);
            node.setSplitable(true);
        } else{
            node.setSplitable(false);
        }
    }

    /**
     * xgboost monotonicity
     * parallel
     * given probs, fill other information
     * @param node
     */
    private static void updateNode(Node node,
                                   RegTreeConfig regTreeConfig,
                                   DataSet dataSet,
                                   double[] labels,
                                   int[] monotonicity) {
        Optional<SplitResult> splitResultOptional = Splitter.split(regTreeConfig,
                dataSet,labels,node.getProbs(), monotonicity);
        if (splitResultOptional.isPresent()){
            SplitResult splitResult = splitResultOptional.get();
            node.setFeatureIndex(splitResult.getFeatureIndex());
            node.setThreshold(splitResult.getThreshold());
            node.setReduction(splitResult.getReduction());
            double leftCount = splitResult.getLeftCount();
            double rightCount = splitResult.getRightCount();
            double totalCount = leftCount + rightCount;
            node.setLeftProb(leftCount/totalCount);
            node.setRightProb(rightCount/totalCount);
            node.setSplitable(true);
        } else{
            node.setSplitable(false);
        }
    }

    private static void cleanLeaves(List<Node> leaves){
        for (Node leaf: leaves){
            leaf.clearProbs();
        }
    }


    /**
     * parallel
     */
    private static void setLeavesOutputs(RegTreeConfig regTreeConfig, List<Node> leaves, LeafOutputCalculator calculator, double[] labels){
        Stream<Node> stream = leaves.stream();
        if (regTreeConfig.isParallel()){
            stream = stream.parallel();
        }

        stream.forEach(leaf -> setLeafOutput(leaf, calculator, labels));
    }

    private static void setLeafOutput(Node leaf, LeafOutputCalculator calculator, double[] labels){
        double[] probs = leaf.getProbs();
        double output = calculator.getLeafOutput(probs, labels);
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
     */
    private static void normalizeReductions(RegressionTree tree, DataSet dataSet){
        int numDataPoints = dataSet.getNumDataPoints();
        List<Node> nodes = tree.traverse();
        for (Node node: nodes){
            double oldReduction = node.getReduction();
            node.setReduction(oldReduction/numDataPoints);
        }
    }
}
