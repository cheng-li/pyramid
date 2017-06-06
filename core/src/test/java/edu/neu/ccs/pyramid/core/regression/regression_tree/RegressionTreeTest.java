package edu.neu.ccs.pyramid.core.regression.regression_tree;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class RegressionTreeTest {

    public static void main(String[] args) {
//        test1();
//        test2();
//        test3();
//        test4();
        test5();
    }

    private static void test1(){
        Node a = new Node();
        a.setFeatureIndex(0);
        a.setThreshold(0.0);
        a.setLeftProb(0.3);
        a.setRightProb(0.7);

        Node b = new Node();
        b.setFeatureIndex(1);
        b.setThreshold(0.1);
        b.setLeftProb(0.8);
        b.setRightProb(0.2);

        Node c = new Node();
        c.setFeatureIndex(2);
        c.setThreshold(0.2);
        c.setLeftProb(0.1);
        c.setRightProb(0.9);

        Node d = new Node();
        d.setLeaf(true);
        d.setValue(1);

        Node e = new Node();
        e.setLeaf(true);
        e.setValue(2);

        Node f = new Node();
        f.setLeaf(true);
        f.setValue(3);

        Node g = new Node();
        g.setLeaf(true);
        g.setValue(4);

        a.setLeftChild(b);
        a.setRightChild(c);
        b.setLeftChild(d);
        b.setRightChild(e);
        c.setLeftChild(f);
        c.setRightChild(g);

        RegressionTree tree = new RegressionTree();
        tree.root = a;
        tree.leaves.add(d);
        tree.leaves.add(e);
        tree.leaves.add(f);
        tree.leaves.add(g);

        Vector vector1 = new DenseVector(3);
        vector1.set(0,Double.NaN);
        vector1.set(1,Double.NaN);
        vector1.set(2,Double.NaN);

        System.out.println(tree.probability(vector1,a));
        System.out.println(tree.probability(vector1,b));
        System.out.println(tree.probability(vector1,c));
        System.out.println(tree.probability(vector1,d));
        System.out.println(tree.probability(vector1,e));
        System.out.println(tree.probability(vector1,f));
        System.out.println(tree.probability(vector1,g));
        System.out.println(tree.predict(vector1));
        System.out.println(0.24+0.06*2+0.07*3+0.63*4);
    }

    private static void test2(){
        Node a = new Node();
        a.setFeatureIndex(0);
        a.setThreshold(0.0);
        a.setLeftProb(0.3);
        a.setRightProb(0.7);

        Node b = new Node();
        b.setFeatureIndex(1);
        b.setThreshold(0.1);
        b.setLeftProb(0.8);
        b.setRightProb(0.2);

        Node c = new Node();
        c.setFeatureIndex(2);
        c.setThreshold(0.2);
        c.setLeftProb(0.1);
        c.setRightProb(0.9);

        Node d = new Node();
        d.setLeaf(true);
        d.setValue(1);

        Node e = new Node();
        e.setLeaf(true);
        e.setValue(2);

        Node f = new Node();
        f.setLeaf(true);
        f.setValue(3);

        Node g = new Node();
        g.setLeaf(true);
        g.setValue(4);

        a.setLeftChild(b);
        a.setRightChild(c);
        b.setLeftChild(d);
        b.setRightChild(e);
        c.setLeftChild(f);
        c.setRightChild(g);

        RegressionTree tree = new RegressionTree();
        tree.root = a;
        tree.leaves.add(d);
        tree.leaves.add(e);
        tree.leaves.add(f);
        tree.leaves.add(g);

        Vector vector1 = new DenseVector(3);
        vector1.set(0,-1);
        vector1.set(1,Double.NaN);
        vector1.set(2,Double.NaN);

        System.out.println(tree.probability(vector1,a));
        System.out.println(tree.probability(vector1,b));
        System.out.println(tree.probability(vector1,c));
        System.out.println(tree.probability(vector1,d));
        System.out.println(tree.probability(vector1,e));
        System.out.println(tree.probability(vector1,f));
        System.out.println(tree.probability(vector1,g));
        System.out.println(tree.predict(vector1));
    }

    private static void test3(){
        Node a = new Node();
        a.setFeatureIndex(0);
        a.setThreshold(0.0);
        a.setLeftProb(0.3);
        a.setRightProb(0.7);

        Node b = new Node();
        b.setFeatureIndex(1);
        b.setThreshold(0.1);
        b.setLeftProb(0.8);
        b.setRightProb(0.2);

        Node c = new Node();
        c.setFeatureIndex(2);
        c.setThreshold(0.2);
        c.setLeftProb(0.1);
        c.setRightProb(0.9);

        Node d = new Node();
        d.setLeaf(true);
        d.setValue(1);

        Node e = new Node();
        e.setLeaf(true);
        e.setValue(2);

        Node f = new Node();
        f.setLeaf(true);
        f.setValue(3);

        Node g = new Node();
        g.setLeaf(true);
        g.setValue(4);

        a.setLeftChild(b);
        a.setRightChild(c);
        b.setLeftChild(d);
        b.setRightChild(e);
        c.setLeftChild(f);
        c.setRightChild(g);

        RegressionTree tree = new RegressionTree();
        tree.root = a;
        tree.leaves.add(d);
        tree.leaves.add(e);
        tree.leaves.add(f);
        tree.leaves.add(g);

        Vector vector1 = new DenseVector(3);
        vector1.set(0,-1);
        vector1.set(1,0.2);
        vector1.set(2,Double.NaN);

        System.out.println(tree.probability(vector1,a));
        System.out.println(tree.probability(vector1,b));
        System.out.println(tree.probability(vector1,c));
        System.out.println(tree.probability(vector1,d));
        System.out.println(tree.probability(vector1,e));
        System.out.println(tree.probability(vector1,f));
        System.out.println(tree.probability(vector1,g));
        System.out.println(tree.predict(vector1));
    }


    private static void test4(){
        Node a = new Node();
        a.setFeatureIndex(0);
        a.setThreshold(0.0);
        a.setLeftProb(0.3);
        a.setRightProb(0.7);

        Node b = new Node();
        b.setFeatureIndex(1);
        b.setThreshold(0.1);
        b.setLeftProb(0.8);
        b.setRightProb(0.2);

        Node c = new Node();
        c.setFeatureIndex(2);
        c.setThreshold(0.2);
        c.setLeftProb(0.1);
        c.setRightProb(0.9);

        Node d = new Node();
        d.setLeaf(true);
        d.setValue(1);

        Node e = new Node();
        e.setLeaf(true);
        e.setValue(2);

        Node f = new Node();
        f.setLeaf(true);
        f.setValue(3);

        Node g = new Node();
        g.setLeaf(true);
        g.setValue(4);

        a.setLeftChild(b);
        a.setRightChild(c);
        b.setLeftChild(d);
        b.setRightChild(e);
        c.setLeftChild(f);
        c.setRightChild(g);

        RegressionTree tree = new RegressionTree();
        tree.root = a;
        tree.leaves.add(d);
        tree.leaves.add(e);
        tree.leaves.add(f);
        tree.leaves.add(g);

        Vector vector1 = new DenseVector(3);
        vector1.set(0,1);
        vector1.set(1,Double.NaN);
        vector1.set(2,Double.NaN);

        System.out.println(tree.probability(vector1,a));
        System.out.println(tree.probability(vector1,b));
        System.out.println(tree.probability(vector1,c));
        System.out.println(tree.probability(vector1,d));
        System.out.println(tree.probability(vector1,e));
        System.out.println(tree.probability(vector1,f));
        System.out.println(tree.probability(vector1,g));
        System.out.println(tree.predict(vector1));


    }

    private static void test5(){
        RegressionTree tree = RegressionTree.newStump(10,0.5,-1.2,3);
        System.out.println(tree);
        Vector vector = new DenseVector(100);
        vector.set(10,0.6);
        System.out.println(tree.predict(vector));
    }
}