package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.List;

/**
 * test lbfgs on synthetic functions
 * Created by chengli on 3/5/15.
 */
public class Exp71 {
    public static void main(String[] args) {
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.DEBUG);
        ctx.updateLoggers();


        Vector bestX = null;
        double min = Double.POSITIVE_INFINITY;
        for (int t = 0;t<100;t++){
            TestFunction3 function = new TestFunction3();
            for (int v=0;v<function.getParameters().size();v++){
                double initial = Sampling.doubleUniform(-10,10);
                function.getParameters().set(v, initial);
            }
            LBFGS lbfgs = new LBFGS(function);
            lbfgs.setEpsilon(0.00000001);
//            for (int i=0;i<100;i++){
////            System.out.println(function.getParameters());
////            System.out.println(function.getValue(function.getParameters()));
////            System.out.println(function.getGradient());
//                lbfgs.iterate();
//            }
            lbfgs.optimize();
            double value = function.getValue(function.getParameters());
            if (value<min){
                min  = value;
                bestX = function.getParameters();
            }
        }

        System.out.println("best x = "+bestX);
        System.out.println("min y = "+min);



//        lbfgs.optimize();


    }

    private static class TestFunction implements Optimizable.ByGradientValue{
        private Vector vector;

        public TestFunction() {
            this.vector = new DenseVector(1);
        }

        public Vector getVector() {
            return vector;
        }

        @Override
        public Vector getGradient() {
            double x = vector.get(0);
            double sum = Math.cos(x)
                    + 6* Math.cos(2 * x + Math.PI)
                    + 15*Math.cos(3*x+0.5*Math.PI)
                    + 8*Math.cos(4*x+0.2*Math.PI)
                    + 1.5*Math.cos(0.5*x+0.4*Math.PI)
                    + 2*Math.cos(0.2*x+0.5*Math.PI);
            Vector gradient = new DenseVector(1);
            gradient.set(0,sum);
            return gradient;
        }

        @Override
        public double getValue(Vector parameters) {
            double x = parameters.get(0);
            double sum = Math.sin(x)
             + 3* Math.sin(2*x+Math.PI)
             + 5*Math.sin(3*x+0.5*Math.PI)
             + 2*Math.sin(4*x+0.2*Math.PI)
             + 3*Math.sin(0.5*x+0.4*Math.PI)
             + 10*Math.sin(0.2*x+0.5*Math.PI);
            return sum;
        }

        @Override
        public Vector getParameters() {
            return vector;
        }

        @Override
        public void refresh() {

        }
    }


    private static class TestFunction2 implements Optimizable.ByGradientValue{
        private Vector vector;

        public TestFunction2() {
            this.vector = new DenseVector(1);
        }

        public Vector getVector() {
            return vector;
        }

        @Override
        public Vector getGradient() {
            double x = vector.get(0);
            double value = 0;
            if (x>0){
                value = 1;
            }
            if (x<0){
                value = -1;
            }
            Vector gradient = new DenseVector(1);
            gradient.set(0,value);
            return gradient;
        }

        @Override
        public double getValue(Vector parameters) {
            double x = parameters.get(0);
            return Math.abs(x);
        }

        @Override
        public Vector getParameters() {
            return vector;
        }

        @Override
        public void refresh() {

        }
    }


    private static class TestFunction3 implements Optimizable.ByGradientValue{
        private Vector vector;

        public TestFunction3() {
            this.vector = new DenseVector(2);
        }



        @Override
        public Vector getGradient() {
            double x1 = vector.get(0);
            double x2 = vector.get(1);
            double g1 = 1*Math.cos(x1)
                    + 0* Math.cos(2 * x2 + Math.PI)
                    + 15*Math.cos(3*x1 + x2 +0.5*Math.PI)
                    + 0*Math.cos(4*x2 +0.2*Math.PI)
                    + 1.5*Math.cos(0.5*x1 + 2*x2 +0.4*Math.PI)
                    + 5*Math.cos(0.2*x1 +0.5*Math.PI);
            double g2 = 0* Math.cos(x1)
                    + 6* Math.cos(2 * x2 + Math.PI)
                    + 5*Math.cos(3*x1 + x2 +0.5*Math.PI)
                    + 8*Math.cos(4*x2 +0.2*Math.PI)
                    + 6*Math.cos(0.5*x1 + 2*x2 +0.4*Math.PI)
                    + 0*Math.cos(0.2*x1 +0.5*Math.PI);
            Vector gradient = new DenseVector(vector.size());
            gradient.set(0,g1);
            gradient.set(1,g2);
            return gradient;
        }

        @Override
        public double getValue(Vector parameters) {
            double x1 = parameters.get(0);
            double x2 = parameters.get(1);
            double sum = Math.sin(x1)
                    + 3* Math.sin(2 * x2 + Math.PI)
                    + 5*Math.sin(3*x1 + x2 +0.5*Math.PI)
                    + 2*Math.sin(4*x2 +0.2*Math.PI)
                    + 3*Math.sin(0.5*x1 + 2*x2 +0.4*Math.PI)
                    + 10*Math.sin(0.2*x1 +0.5*Math.PI);
            return sum;
        }

        @Override
        public Vector getParameters() {
            return vector;
        }

        @Override
        public void refresh() {

        }
    }
}
