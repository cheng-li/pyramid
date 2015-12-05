package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

/**
 * Created by Rainicy on 11/28/15.
 */
public class DynamicProgrammingTest {

    public static void main(String[] args) throws Exception {
        test();
    }

    private static void test() throws Exception{

        double[][] probs = new double[][]{ {0.8, 0.2}, {0.6, 0.4}, {0.1, 0.9}};
        double[][] logProbs = new double[probs.length][probs[0].length];
        for (int l=0; l<probs.length; l++) {
            for (int i=0; i<probs[l].length; i++) {
                logProbs[l][i] = Math.log(probs[l][i]);
            }
        }

        DynamicProgramming dp = new DynamicProgramming(probs,logProbs);

        for (int i=0; i<8; i++) {
            System.out.print("i: " + i);
            System.out.print("\tprob: " + String.format("%.3f", dp.highestProb()));
            System.out.print("\ty: " + dp.nextHighest());
            System.out.println("\tQ: " + dp.dp);
        }
    }
}

