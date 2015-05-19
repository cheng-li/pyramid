package edu.neu.ccs.pyramid.experiment;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by Rainicy on 5/18/15.
 */
public class Exp200 {

//    public static void main(String[] args) {
//        if (args.length !=1){
//            throw new IllegalArgumentException("please specify the config file");
//        }
//
//        Config config = new Config(args[0]);
//        System.out.println(config);
//

//    }

//    public static void main(String[] args) {
//        Vector vector1 = new DenseVector(10000000);
//        Vector vector2 = new RandomAccessSparseVector(10000000);
//        StopWatch stopWatch = new StopWatch();
//        stopWatch.start();
//        for (int i=0;i<vector1.size();i++){
//            vector1.get(i);
//        }
//        System.out.println(stopWatch);
//
//        stopWatch.reset();
//        stopWatch.start();
//        for (Vector.Element element: vector1.all()){
//            int index = element.index();
//            double value = element.get();
//            }
//        System.out.println(stopWatch);
//
//        stopWatch.reset();
//        stopWatch.start();
//        for (Vector.Element element: vector2.all()){
//            int index = element.index();
//            double value = element.get();
//        }
//        System.out.println(stopWatch);
//    }

    public static void main(String[] args) {
        Vector vector1 = new DenseVector(10000000);
        Vector vector2 = new RandomAccessSparseVector(10000000);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int i=0;i<vector1.size();i++){
            vector1.get(i);
            }
        System.out.println(stopWatch);

        stopWatch.reset();
        stopWatch.start();
        for (Vector.Element element: vector1.all()){
            int index = element.index();
            double value = element.get();
            }
        System.out.println(stopWatch);

        stopWatch.reset();
        stopWatch.start();
        for (Vector.Element element: vector2.all()){
            int index = element.index();
            double value = element.get();
            }
        System.out.println(stopWatch);

        stopWatch.reset();
        stopWatch.start();
        for (Vector.Element element: vector2.nonZeroes()){
            int index = element.index();
            double value = element.get();
            }
        System.out.println(stopWatch);
    }
}
