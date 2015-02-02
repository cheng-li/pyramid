package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

import java.io.Serializable;

import java.util.*;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * Created by chengli on 8/6/14.
 */
public class RegressionTree implements Regressor, Serializable {

    private static final long serialVersionUID = 2L;

    /**
     * including intermediate nodes
     */
    int numNodes;

    protected Node root;
    /**
     * the actual number of leaves may be smaller than maxNumLeaves
     */

    protected List<Node> leaves;

    protected RegressionTree() {
        this.numNodes = 0;
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

    @Override
    public double predict(Vector vector){
        // use as a simple cache
        int numNodes = this.numNodes;
        boolean[] calculated = new boolean[numNodes];
        double[] probs = new double[numNodes];
        double prediction = 0;
        for (Node leaf: this.leaves){
            double prob = probability(vector,leaf, calculated, probs);
            prediction += prob*leaf.getValue();
        }
        return prediction;
    }


    /**
     * the probability of a vector falling into the node
     * cache probabilities
     * @param vector
     * @param node
     * @return
     */
     double probability(Vector vector, Node node, boolean[] calculated, double[] probs){
        int id = node.getId();

        if (calculated[id]){
            return probs[id];
        }

        if (node == root){
            return 1;
        }

        Node parent = node.getParent();
        int featureIndex = parent.getFeatureIndex();
        double threshold = parent.getThreshold();
        boolean isLeftChild = (node==parent.getLeftChild());
        double featureValue = vector.get(featureIndex);


        // for missing value
        if (Double.isNaN(featureValue)){
            if (isLeftChild){
                double prob = parent.getLeftProb()*probability(vector,parent, calculated, probs);
                calculated[id] = true;
                probs[id] = prob;
                return prob;
            } else {
                double prob = parent.getRightProb()*probability(vector,parent, calculated, probs);
                calculated[id] = true;
                probs[id] = prob;
                return prob;
            }
        }

        // for existing value

        if (isLeftChild && featureValue <= threshold){
            double prob = probability(vector,parent,calculated, probs);
            calculated[id] = true;
            probs[id] = prob;
            return prob;
        }

        if (isLeftChild && featureValue > threshold){
            double prob = 0;
            calculated[id] = true;
            probs[id] = prob;
            return prob;
        }

        if (!isLeftChild && featureValue <= threshold){
            double prob = 0;
            calculated[id] = true;
            probs[id] = prob;
            return prob;
        }

        if (!isLeftChild && featureValue > threshold){
            double prob = probability(vector,parent, calculated, probs);
            calculated[id] = true;
            probs[id] = prob;
            return prob;
        }

        // this should not happen
        return 1;
    }


    /**
     * the probability of a vector falling into the node
     * doesn't cache probabilities
     * @param vector
     * @param node
     * @return
     */
    double probability(Vector vector, Node node){
        int id = node.getId();

        if (node == root){
            return 1;
        }

        Node parent = node.getParent();
        int featureIndex = parent.getFeatureIndex();
        double threshold = parent.getThreshold();
        boolean isLeftChild = (node==parent.getLeftChild());
        double featureValue = vector.get(featureIndex);

        // for missing value
        if (Double.isNaN(featureValue)){
            if (isLeftChild){
                double prob = parent.getLeftProb()*probability(vector,parent);
                return prob;
            } else {
                double prob = parent.getRightProb()*probability(vector,parent);
                return prob;
            }
        }

        // for existing value
        if (isLeftChild && featureValue <= threshold){
            double prob = probability(vector,parent);
            return prob;
        }

        if (isLeftChild && featureValue > threshold){
            double prob = 0;
            return prob;
        }

        if (!isLeftChild && featureValue <= threshold){
            double prob = 0;
            return prob;
        }

        if (!isLeftChild && featureValue > threshold){
            double prob = probability(vector,parent);
            return prob;
        }

        // this should not happen
        return 1;
    }



//    public DecisionProcess getDecisionProcess(float [] vector,List<Feature> features){
//        StringBuilder sb = new StringBuilder();
//        Node nodeToCheck = this.root;
//        while (! this.leaves.contains(nodeToCheck)){
//            int featureIndex = nodeToCheck.getFeatureIndex();
//            float threshold = nodeToCheck.getThreshold();
//            if (vector[featureIndex] <= threshold){
//                nodeToCheck = nodeToCheck.getLeftChild();
//                sb.append(features.get(featureIndex).getFeatureName());
//                sb.append("(").append(vector[featureIndex]).append("<=").append(threshold).append(")  ");
//            }else{
//                nodeToCheck = nodeToCheck.getRightChild();
//                sb.append(features.get(featureIndex).getFeatureName());
//                sb.append("(").append(vector[featureIndex]).append(">").append(threshold).append(")  ");
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
