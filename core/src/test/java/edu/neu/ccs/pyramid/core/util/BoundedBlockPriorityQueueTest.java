package edu.neu.ccs.pyramid.core.util;

import edu.neu.ccs.pyramid.core.feature.Ngram;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 2/7/17.
 */
public class BoundedBlockPriorityQueueTest {
    public static void main(String[] args) {
        test2();



    }

    private static void test1(){
        Comparator<Double> comparator = Comparator.comparing(d->d.doubleValue());
        BoundedBlockPriorityQueue<Double> queue = new BoundedBlockPriorityQueue<>(3, comparator);

        List<Double> all = new ArrayList<>();
        for (int i=0;i<10;i++){
            all.add(Math.random());
        }

        System.out.println(all);

//        for (double d: all){
//            queue.add(d);
//        }
        all.stream()
                .parallel()
                .forEach(d->queue.add(d));

        Collections.sort(all);

        System.out.println("sorted = "+ all);

        System.out.println(queue);
    }


    private static void test2(){
        Comparator<Pair<Ngram, Double>> comparator = Comparator.comparing(p->p.getSecond());
        BoundedBlockPriorityQueue<Pair<Ngram, Double>> queue = new BoundedBlockPriorityQueue<>(3, comparator);

        List<Double> all = new ArrayList<>();
        for (int i=0;i<10;i++){
            all.add(Math.random());
        }

        System.out.println(all);

//        for (double d: all){
//            queue.add(d);
//        }
        IntStream.range(0,all.size())
                .parallel()
                .forEach(d->{
                    Ngram ngram = new Ngram();
                    ngram.setNgram(""+d);
                    queue.add(new Pair<>(ngram, all.get(d)));
                });

        Collections.sort(all);

        System.out.println("sorted = "+ all);

        System.out.println(queue);
    }
}