package edu.neu.ccs.pyramid.core.multilabel_classification;


/**
 * Created by Rainicy on 11/28/15.
 */
public class DynamicProgrammingTest {

    public static void main(String[] args) throws Exception {
        test3();
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
            System.out.print("\tprob: " + String.format("%.3f", dp.nextHighestProb()));
            System.out.print("\ty: " + dp.nextHighestVector());
            System.out.println("\tQ: " + dp.getQueue());
        }
    }

    private static void test2() throws Exception{

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
            System.out.print("\t" + dp.nextHighest());
            System.out.println("\tQ: " + dp.getQueue());
        }
    }

    private static void test3() throws Exception{

        double[] prob = {0.2,0.4,0.9};

        DynamicProgramming dp = new DynamicProgramming(prob);

        for (int i=0; i<8; i++) {
            System.out.print("i: " + i);
            System.out.print("\t"+dp.nextHighest());
            System.out.println("\tQ: " + dp.getQueue());
        }
    }
}

