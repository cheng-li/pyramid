package edu.neu.ccs.pyramid.core.eval;

public class RecallTest {
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        int[] predictions = {0,0,1,1,1};
        int[] labels = {0,0,0,1,1};
        System.out.println(Recall.recall(labels,predictions,0));
    }
}