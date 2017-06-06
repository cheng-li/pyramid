package edu.neu.ccs.pyramid.core.util;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * The code is based on the code from https://evilzone.org/java/%28java-snippet%29-cli-animated-progress-bar/msg55871/?PHPSESSID=494195005cd1f5106876ea2b5e19b120#msg55871
 *
 * Created by chengli on 2/9/16.
 */
public class ProgressBar {
    private String barStart = "[";
    private String barEnd = "]";
    private String arrowBody = "=";
    private String arrowEnd = ">";
    private int width = 50;
    private AtomicInteger currentCount = new AtomicInteger(0);
    private int total;

    public ProgressBar(int total) {
        this.total = total;
    }

    public synchronized void incrementAndPrint(){
        int current = currentCount.incrementAndGet();
        if (Math.floor(((double)current)/total*100) > Math.floor(((double)current-1)/total*100)){
           printProcessBar((int)Math.floor(((double)current)/total*100));
        }
    }

    private void printProcessBar(int percent) {
        int processWidth = percent * width / 100;
        System.out.print("\r" + barStart);
        for (int i = 0; i < processWidth; i++) {
            System.out.print(arrowBody);
        }
        System.out.print(arrowEnd);
        for (int i = processWidth; i < width; i++) {
            System.out.print(" ");
        }
        System.out.print(barEnd + percent + "%");
    }


}
