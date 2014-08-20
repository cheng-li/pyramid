package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.FeatureRow;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.DenseVector;
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




    public double predict(FeatureRow featureRow){
        return predict(featureRow.getVector());
    }

    private double predict(Vector vector){
        return predict(vector, this.root);
    }

    private double predict(Vector vector, Node node){
        if (node.isLeaf()){
            return node.getValue();
        } else if (vector.get(node.getFeatureIndex())<=node.getThreshold()){
            return predict(vector,node.getLeftChild());
        } else {
            return predict(vector,node.getRightChild());
        }
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
                        sb.append(node1.getFeatureIndex()).append("<=").append(node1.getThreshold()).append("   ");
                    } else {
                        sb.append(node1.getFeatureIndex()).append(">").append(node1.getThreshold()).append("   ");
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

//    public Set<String> getSkipNgramNames(List<Feature> features){
//        Set<String> names = new HashSet<String>();
//        List<Integer> indices = this.getFeatureIndices();
//        for (int i:indices){
//            Feature ngram = features.get(i);
//            if (((Ngram)ngram).getNumTerms()>=2){
//                names.add(ngram.getFeatureName());
//            }
//        }
//        return names;
//    }

//    public String getRootFeatureName(List<Feature> features){
//        int featureIndex= this.root.getFeatureIndex();
//        return features.get(featureIndex).getFeatureName();
//    }

    public int getRootFeatureIndex(){
        return this.root.getFeatureIndex();
    }

    public double getRootRightOutput(){
        return this.root.getRightChild().getValue();
    }

    public double getRootReduction(){
        return this.root.getReduction();
    }





}
