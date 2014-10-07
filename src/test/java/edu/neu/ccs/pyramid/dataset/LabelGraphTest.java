package edu.neu.ccs.pyramid.dataset;

import java.util.List;

import static org.junit.Assert.*;

public class LabelGraphTest {

    public static void main(String[] args) {

        test1();
        test2();
        test3();
        test4();
        test5();

    }

    static void test1() {

        int numLabels = 4;
        LabelGraph graph = new LabelGraph(numLabels);
        graph.parser("1 hierarchy 2,3");
        graph.parser("0 exclusive 1");
        graph.parser("destination 0,2,3");

        System.out.println("Ancestors: " + graph.getAncestorLabels(3).toString());
        System.out.println("Descendants: " + graph.getDescendantLabels(1).toString());
        System.out.println("Exclusions: " + graph.getExclusiveLabels(1).toString());

    }

    static void test2() {

        int numLabels = 4;
        LabelGraph graph = new LabelGraph(numLabels);
        graph.parser("1 hierarchy 2,3");
        graph.parser("0 exclusive 1");
        graph.parser("destination 0,2,3");

        System.out.println("Consistency: " + graph.isConsistent());

    }

    static void test3() {

        int numLabels = 4;
        LabelGraph graph = new LabelGraph(numLabels);
        graph.parser("1 hierarchy 2,3");
        graph.parser("0 exclusive 1");
        graph.parser("destination 0,2,3");

        if (!graph.isHierarchySubGraphDag()) {
            System.out.println("Graph is not a DAG!");
        }
        else {
            //test4
            List<MultiLabel> assignments = graph.getLegalAssignments();
            System.out.println("Assignments: ");
            for (int i = 0; i < assignments.size(); i++) {
                System.out.println(assignments.get(i).toString());
            }
        }

    }

    static void test4() {

        int numLabels = 4;
        LabelGraph graph1 = new LabelGraph(numLabels);
        graph1.parser("1 hierarchy 2,3");
        graph1.parser("0 exclusive 1");
        graph1.parser("destination 0,2,3");
        LabelGraph graph2 = new LabelGraph(numLabels);
        graph2.parser("0 exclusive 1");
        graph2.parser("1 hierarchy 2,3");
        graph2.parser("destination 0,2,3");
        System.out.println(LabelGraph.isEquivalent(graph1, graph2));

    }

    static void test5() {

        int numLabels = 4;
        LabelGraph graph = new LabelGraph(numLabels);
        graph.parser("1 hierarchy 2,3");
        graph.parser("2 hier 3");
        graph.parser("0 exclusive 1");
        graph.parser("destination 0,2,3");

        System.out.println(graph.isMinimallySparse());
        System.out.println(graph.isMaximallyDense());
        LabelGraph.sparsify(graph);
        System.out.println(graph.isMinimallySparse());

    }
}