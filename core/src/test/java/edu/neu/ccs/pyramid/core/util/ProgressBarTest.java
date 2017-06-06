package edu.neu.ccs.pyramid.core.util;


import java.util.stream.IntStream;

public class ProgressBarTest  {
    public static void main(String[] args) throws Exception{
        test4();
    }


    private static void test1() throws Exception{
        ProgressBar bar = new ProgressBar(100);
        for (int i = 0; i < 100; i++) {
            Thread.sleep(200);
            bar.incrementAndPrint();
        }

    }

    private static void test2() throws Exception{
        ProgressBar bar = new ProgressBar(150);
        for (int i = 0; i < 100; i++) {
            Thread.sleep(200);
            bar.incrementAndPrint();
        }

    }

    private static void test3() throws Exception{
        ProgressBar bar = new ProgressBar(150);
        IntStream.range(0,100).forEach(i->{
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            bar.incrementAndPrint();
        });
    }

    private static void test4() throws Exception{
        ProgressBar bar = new ProgressBar(100);
        IntStream.range(0,100).parallel().forEach(i->{
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            bar.incrementAndPrint();
        });
    }

}