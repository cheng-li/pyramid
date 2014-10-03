package edu.neu.ccs.pyramid.eval;

import static org.junit.Assert.*;

public class MicroAveragedMeasuresTest {
    public static void main(String[] args) {
        test1();
    }

    /**
     * follow the results in
     * http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
     */
    private static void test1(){
        int[] labels = {0, 1, 2, 0, 1, 2};
        int[] prediction = {0, 2, 1, 0, 0, 1};
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(3,labels,prediction);
        System.out.println(confusionMatrix.printWithIntLabels());
        System.out.println(new PerClassMeasures(confusionMatrix,0));
        System.out.println(new PerClassMeasures(confusionMatrix,1));
        System.out.println(new PerClassMeasures(confusionMatrix,2));
        MicroAveragedMeasures measures = new MicroAveragedMeasures(confusionMatrix);
        System.out.println(measures);
    }

}