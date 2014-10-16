package edu.neu.ccs.pyramid.regression.prob_reg_tree;

import edu.neu.ccs.pyramid.dataset.FeatureRow;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

import java.io.Serializable;

import java.util.*;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * Created by chengli on 8/6/14.
 */
public class RegressionTree implements Regressor, Serializable {

    private static final long serialVersionUID = 1L;

    protected Node root;
    /**
     * the actual number of leaves may be smaller than maxNumLeaves
     */

    protected List<Node> leaves;

    protected RegressionTree() {
        this.leaves = new ArrayList<>();
    }

    /**
     *
     * @return number of leaves
     */
    public int getNumLeaves(){
        return leaves.size();
    }


    public Node getRoot() {
        return root;
    }

    public double predict(FeatureRow featureRow){
        return predict(featureRow.getVector());
    }

    double predict(Vector vector){
        double prediction = 0;
        for (Node leaf: this.leaves){
            double prob = probability(vector,leaf);
            prediction += prob*leaf.getValue();
        }
        return prediction;
    }


    /**
     * the probability of a vector falling into the leaf
     * @param vector
     * @param node
     * @return
     */
     double probability(Vector vector, Node node){
        if (node == root){
            return 1;
        }
        Node parent = node.getParent();
        int featureIndex = parent.getFeatureIndex();
        double threshold = parent.getThreshold();
        boolean isLeftChild = (node==parent.getLeftChild());

        // for missing value
        if (Double.isNaN(vector.get(featureIndex))){
            if (isLeftChild){
                return parent.getLeftProb()*probability(vector,parent);
            } else {
                return parent.getRightProb()*probability(vector,parent);
            }
        }

        // for existing value
        double value = vector.get(featureIndex);
        if (isLeftChild && value <= threshold){
            return probability(vector,parent);
        }

        if (isLeftChild && value > threshold){
            return 0;
        }

        if (!isLeftChild && value <= threshold){
            return 0;
        }

        if (!isLeftChild && value > threshold){
            return probability(vector,parent);
        }

        // this should not happen
        return 1;
    }

//    public DecisionProcess getDecisionProcess(float [] featureRow,List<Feature> features){
//        StringBuilder sb = new StringBuilder();
//        Node nodeToCheck = this.root;
//        while (! this.leaves.contains(nodeToCheck)){
//            int featureIndex = nodeToCheck.getFeatureIndex();
//            float threshold = nodeToCheck.getThreshold();
//            if (featureRow[featureIndex] <= threshold){
//                nodeToCheck = nodeToCheck.getLeftChild();
//                sb.append(features.get(featureIndex).getFeatureName());
//                sb.append("(").append(featureRow[featureIndex]).append("<=").append(threshold).append(")  ");
//            }else{
//                nodeToCheck = nodeToCheck.getRightChild();
//                sb.append(features.get(featureIndex).getFeatureName());
//                sb.append("(").append(featureRow[featureIndex]).append(">").append(threshold).append(")  ");
//            }
//        }
//        return new DecisionProcess(sb.toString(),nodeToCheck.getValue());
//
//    }

    public List<Integer> getFeatureIndices(){
        List<Integer> featureIndices = new ArrayList<Integer>();
        LinkedBlockingDeque<Node> queue = new LinkedBlockingDeque<Node>();
        queue.offer(this.root);
        while(queue.size()!=0){
            Node node = queue.poll();
            if (! node.isLeaf()){
                //don't add the feature for leaf node, as it is useless
                featureIndices.add(node.getFeatureIndex());
                queue.offer(node.getLeftChild());
                queue.offer(node.getRightChild());
            }
        }
        return featureIndices;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Node node: this.leaves){
            Stack<Node> stack = new Stack<Node>();
            while(true){
                stack.push(node);
                if (node.getParent()==null){
                    break;
                }
                node = node.getParent();
            }
            while(!stack.empty()){
                Node node1 = stack.pop();
                if (!node1.isLeaf()){
                    Node node2 = stack.peek();
                    if (node2 == node1.getLeftChild()){
                        sb.append(node1.getFeatureIndex()).append("(")
                                .append(node1.getFeatureName()).append(")")
                                .append("<=").append(node1.getThreshold()).append("   ");
                    } else {
                        sb.append(node1.getFeatureIndex()).append("(")
                                .append(node1.getFeatureName()).append(")").
                                append(">").append(node1.getThreshold()).append("   ");
                    }
                } else{
                    sb.append(": ").append(node1.getValue()).append("\n");
                }
            }
        }
        return sb.toString();
    }

//    public String display(List<Feature> features){
//        StringBuilder sb = new StringBuilder();
//        for (Node node: this.leaves){
//            Stack<Node> stack = new Stack<Node>();
//            while(true){
//                stack.push(node);
//                if (node.getParent()==null){
//                    break;
//                }
//                node = node.getParent();
//            }
//            while(!stack.empty()){
//                Node node1 = stack.pop();
//                if (!node1.isLeaf()){
//                    Node node2 = stack.peek();
//                    if (node2 == node1.getLeftChild()){
//                        sb.append("feature "+node1.getFeatureIndex()+"("+features.get(node1.getFeatureIndex()).getFeatureName()+")"+"<="+node1.getThreshold()+"   ");
//                    } else {
//                        sb.append("feature "+node1.getFeatureIndex()+"("+features.get(node1.getFeatureIndex()).getFeatureName()+")"+">"+node1.getThreshold()+"   ");
//                    }
//                } else{
//                    sb.append(": "+node1.getValue()+"\n");
//                }
//            }
//        }
//        return sb.toString();
//    }






    /**
     * pre-order traverse
     * http://en.wikipedia.org/wiki/Tree_traversal
     * no duplicates
     * @return all nodes
     */
    List<Node> traverse(){
        List<Node> list = new ArrayList<>();
        Deque<Node> deque = new ArrayDeque<>();
        deque.addFirst(this.root);
        while(deque.size()!=0){
            Node visit = deque.removeFirst();
            list.add(visit);
            if (!visit.isLeaf()){
                deque.addFirst(visit.getRightChild());
                deque.addFirst(visit.getLeftChild());
            }
        }
        return list;
    }

}
