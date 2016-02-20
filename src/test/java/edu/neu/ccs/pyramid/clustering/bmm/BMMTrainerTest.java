package edu.neu.ccs.pyramid.clustering.bmm;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import static org.junit.Assert.*;

public class BMMTrainerTest {

    public static void main(String[] args) {
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.DEBUG);
        ctx.updateLoggers();
        test5();

    }

    private static void test1(){
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numFeatures(1).numDataPoints(3)
                .dense(true)
                .build();

        dataSet.setFeatureValue(0,0,1);
        dataSet.setFeatureValue(1,0,1);

        System.out.println("dataset = "+dataSet);
        BMMTrainer trainer = new BMMTrainer(dataSet,1);
        System.out.println(trainer.bmm);
        BMM bmm = trainer.train();
        System.out.println(bmm);
    }


    private static void test2(){
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numFeatures(1).numDataPoints(2)
                .dense(true)
                .build();

        dataSet.setFeatureValue(0,0,1);


        System.out.println("dataset = "+dataSet);
        BMMTrainer trainer = new BMMTrainer(dataSet,1);
        System.out.println(trainer.bmm);
        BMM bmm = trainer.train();
        System.out.println(bmm);
    }

    private static void test3(){
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numFeatures(1).numDataPoints(2)
                .dense(true)
                .build();

        dataSet.setFeatureValue(0,0,1);


        System.out.println("dataset = "+dataSet);
        BMMTrainer trainer = new BMMTrainer(dataSet,2);
        System.out.println(trainer.bmm);
        BMM bmm = trainer.train();
        System.out.println(bmm);
    }

    private static void test4(){
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numFeatures(2).numDataPoints(10)
                .dense(true)
                .build();

        for (int i=0;i<5;i++){
            dataSet.setFeatureValue(i,0,1);
        }

        for (int i=5;i<10;i++){
            dataSet.setFeatureValue(i,1,1);
        }



        System.out.println("dataset = "+dataSet);
        BMMTrainer trainer = new BMMTrainer(dataSet,2);
        System.out.println(trainer.bmm);
        trainer.train();
//        for (int iter=0;iter<100;iter++){
//            trainer.iterate();
//        }

        System.out.println(trainer.bmm);
    }


    private static void test5(){
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numFeatures(5).numDataPoints(20)
                .dense(true)
                .build();

        for (int i=0;i<5;i++){
            dataSet.setFeatureValue(i,0,1);
        }

        for (int i=5;i<10;i++){
            dataSet.setFeatureValue(i,1,1);
        }

        for (int i=10;i<20;i++){
            dataSet.setFeatureValue(i,2,1);
            dataSet.setFeatureValue(i,3,1);
        }



        System.out.println("dataset = "+dataSet);
        BMMTrainer trainer = new BMMTrainer(dataSet,3);
        System.out.println(trainer.bmm);
        trainer.train();
//        for (int iter=0;iter<100;iter++){
//            trainer.iterate();
//        }

        System.out.println(trainer.bmm);
        Vector vector1= new DenseVector(5);
        vector1.set(0,1);

        Vector vector2= new DenseVector(5);
        vector2.set(1,1);

        Vector vector3= new DenseVector(5);
        vector3.set(2,1);
        vector3.set(3,1);

        System.out.println(Math.exp(trainer.bmm.logProbability(vector1)));
        System.out.println(Math.exp(trainer.bmm.logProbability(vector2)));
        System.out.println(Math.exp(trainer.bmm.logProbability(vector3)));

    }


    private static void test6(){
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numFeatures(5).numDataPoints(20)
                .dense(true)
                .build();

        for (int i=0;i<5;i++){
            dataSet.setFeatureValue(i,0,1);
        }

        for (int i=5;i<10;i++){
            dataSet.setFeatureValue(i,1,1);
        }

        for (int i=10;i<20;i++){
            dataSet.setFeatureValue(i,1,1);
            dataSet.setFeatureValue(i,2,1);
            dataSet.setFeatureValue(i,3,1);
        }



        System.out.println("dataset = "+dataSet);
        BMMTrainer trainer = new BMMTrainer(dataSet,3);
        System.out.println(trainer.bmm);
        BMM bmm = trainer.train();
//        for (int iter=0;iter<100;iter++){
//            trainer.iterate();
//        }

        System.out.println(bmm);

        for (int i=0;i<5;i++){
            System.out.println("sample "+i);
            System.out.println(bmm.sample());
        }

    }

}