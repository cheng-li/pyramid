package edu.neu.ccs.pyramid.feature_extraction;

import static org.junit.Assert.*;

public class KaryTreeTest {
    public static void main(String[] args) {
        KaryTree<Integer> tree = new KaryTree<>();
        tree.root.setValue(0);
        Node<Integer> b = new Node<>();
        b.setValue(1);
        Node<Integer> c = new Node<>();
        c.setValue(2);
        Node<Integer> d = new Node<>();
        d.setValue(2);
        Node<Integer> e = new Node<>();
        e.setValue(3);
        Node<Integer> f = new Node<>();
        f.setValue(3);
        Node<Integer> g = new Node<>();
        g.setValue(4);

        tree.root.addChild(b);
        tree.root.addChild(c);
        b.addChild(d);
        b.addChild(e);
        c.addChild(f);
        c.addChild(g);
        tree.leaves.add(d);
        tree.leaves.add(e);
        tree.leaves.add(f);
        tree.leaves.add(g);

        System.out.println(tree.getAllPaths());

    }

}