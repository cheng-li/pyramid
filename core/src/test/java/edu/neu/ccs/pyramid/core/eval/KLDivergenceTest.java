package edu.neu.ccs.pyramid.core.eval;

public class KLDivergenceTest {
    public static void main(String[] args) {
        test2();
    }

    private static void test1(){
        double[] a= {0.0, 1.0};
        double[] b= {1.0, 5.1875766101901845E-104};
        System.out.println(KLDivergence.kl(a,b));

    }

    private static void test2(){
        double[] a= {0.9657857533872317, 0.034214246612768354};
        double[] b= {0.0, 1.0};
        System.out.println(KLDivergence.kl(a,b));

    }

}