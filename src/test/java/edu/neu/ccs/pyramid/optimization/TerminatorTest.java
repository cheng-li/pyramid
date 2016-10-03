package edu.neu.ccs.pyramid.optimization;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

public class TerminatorTest {
    public static void main(String[] args) {
//        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
//        Configuration config = ctx.getConfiguration();
//        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
//        loggerConfig.setLevel(Level.DEBUG);
//        ctx.updateLoggers();
        test4();

    }

    private static void test1(){
        double[] values = {1,2,-1,4,5,5.1,5.11,5.11,3,3.1,3.11,3.09,3.08,3.07,3.08};
        Terminator checker = new Terminator();
        for (double value: values){
            checker.add(value);
//            System.out.println("iteration " + checker.getNumIterations());
//            System.out.println("his="+checker.getHistory());
//            System.out.println("min="+checker.getMinValue());
//            System.out.println("max="+checker.getMaxValue());
//            System.out.println("stable="+checker.getStableIterations());
//            System.out.println("converge="+checker.isConverged());
        }
    }


    private static void test2(){
        double[] values = {1,2,-1,4,5,5.1,5.11,5.11,3,3.1,3.11,3.09,3.08,3.07,3.08};
        Terminator checker = new Terminator();
        checker.setMaxIteration(100).setMode(Terminator.Mode.FINISH_MAX_ITER);
        for (double value: values){
            checker.add(value);
//            System.out.println("iteration " + checker.getNumIterations());
//            System.out.println("his="+checker.getHistory());
//            System.out.println("min="+checker.getMinValue());
//            System.out.println("max="+checker.getMaxValue());
//            System.out.println("stable="+checker.getStableIterations());
//            System.out.println("converge="+checker.isConverged());
        }
    }

    private static void test3(){
        Terminator terminator = new Terminator();
        for (int i=0;i<10;i++){
            terminator.add(0);
            System.out.println("i="+i);
            System.out.println(terminator.shouldTerminate());
        }
    }


    private static void test4(){
        Terminator terminator = new Terminator();
        for (int i=0;i<10;i++){
            terminator.add(-1);
            System.out.println("i="+i);
            System.out.println(terminator.shouldTerminate());
        }
    }


}