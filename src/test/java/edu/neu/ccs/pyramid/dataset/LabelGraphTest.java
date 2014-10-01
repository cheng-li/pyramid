package edu.neu.ccs.pyramid.dataset;

import java.util.List;

import static org.junit.Assert.*;

public class LabelGraphTest {


    public static void main(String[] args) {
        int numLabels = 7;
        LabelGraph graph = new LabelGraph(numLabels);
        graph.setHierarchyEdge(0, 1);
        graph.setHierarchyEdge(0, 2);
        graph.setHierarchyEdge(3, 4);
        graph.setExclusionEdge(0, 3);
        graph.setExclusionEdge(3, 5);
        graph.setExclusionEdge(4, 6);
        graph.setHierarchyEdge(4, 6);
        graph.setHierarchyEdge(6, 3);
        //test1
        System.out.println("Ancestors: " + graph.getAncestorLabels(4).toString());
        System.out.println("Descendants: " + graph.getDescendantLabels(0).toString());
        System.out.println("Exclusions: " + graph.getExclusiveLabels(5).toString());
        System.out.println("Overlapping: " + graph.getOverlappingLabels(2).toString());  //?1
        //test2
        System.out.println("Consistency: " + graph.isConsistent());
        //test3
        if (!graph.isHierarchySubGraphDag()) {
            System.out.println("Graph is not a DAG!");
        }
        else {
            //test4
            List<boolean[]> assignments = graph.getLegalAssignments();
            System.out.println("Assignments: ");
            for (int i = 0; i < assignments.size(); i++) {
                for (int j = 0; j < numLabels; j++) {
                    if (assignments.get(i)[j] == true) {
                        System.out.print(j + " ");
                    }
                }
                System.out.println();
            }
        }
    }
}