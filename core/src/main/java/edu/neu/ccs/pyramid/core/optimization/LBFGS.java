package edu.neu.ccs.pyramid.core.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Iterator;
import java.util.LinkedList;


/**
 * Numerical Optimization, Second Edition, Jorge Nocedal Stephen J. Wright
 * Algorithm 7.4 and 7.5
 * Liu, Tao-Wen.
 * "A regularized limited memory BFGS method for nonconvex unconstrained minimization."
 * Numerical Algorithms 65.2 (2014): 305-323.
 * Formula 2.7
 * Created by chengli on 12/9/14.
 */
public class LBFGS extends GradientValueOptimizer implements Optimizer{
    private static final Logger logger = LogManager.getLogger();
    private BackTrackingLineSearcher lineSearcher;
    /**
     * history length;
     */
    private double m = 5;
    private LinkedList<Vector> sQueue;
    private LinkedList<Vector> yQueue;
    private LinkedList<Double> rhoQueue;



    public LBFGS(Optimizable.ByGradientValue function) {
        super(function);
        this.lineSearcher = new BackTrackingLineSearcher(function);
        lineSearcher.setInitialStepLength(1);
        this.sQueue = new LinkedList<>();
        this.yQueue = new LinkedList<>();
        this.rhoQueue = new LinkedList<>();

    }


    public void iterate(){
        if (logger.isDebugEnabled()){
            logger.debug("start one iteration");
        }

        // we need to make a copy of the gradient; should not use pointer
        Vector oldGradient = new DenseVector(function.getGradient());
        Vector direction = findDirection();
        if (logger.isDebugEnabled()){
            logger.debug("norm of direction = "+direction.norm(2));
        }
        BackTrackingLineSearcher.MoveInfo moveInfo = lineSearcher.moveAlongDirection(direction);

        Vector s = moveInfo.getStep();
        Vector newGradient = function.getGradient();
        Vector y = newGradient.minus(oldGradient);
        double denominator = y.dot(s);

        //todo what to do if denominator is not positive?

        double rho = 0;
        if (denominator>0){
            rho = 1/denominator;
        }
        else {
            terminator.forceTerminate();
            if (logger.isWarnEnabled()){
                logger.warn("denominator <= 0");
            }
        }


        if (logger.isDebugEnabled()){
            if (y.size()<100){
                logger.debug("y= "+y);
                logger.debug("s= " + s);
            }
            logger.debug("denominator = "+denominator);
            logger.debug("rho = "+rho);
        }
        sQueue.add(s);
        yQueue.add(y);
        rhoQueue.add(rho);
        if (sQueue.size()>m){
            sQueue.remove();
            yQueue.remove();
            rhoQueue.remove();
        }
        if (logger.isDebugEnabled()){
            logger.debug("finish one iteration");
        }
        terminator.add(function.getValue());
    }

    Vector findDirection(){
        Vector g = function.getGradient();
        //using dense vector is much faster
        Vector q = new DenseVector(g.size());
        q.assign(g);
        Iterator<Double> rhoDesIterator = rhoQueue.descendingIterator();
        Iterator<Vector> sDesIterator = sQueue.descendingIterator();
        Iterator<Vector> yDesIterator = yQueue.descendingIterator();

        LinkedList<Double> alphaQueue = new LinkedList<>();

        while(rhoDesIterator.hasNext()){
            double rho = rhoDesIterator.next();
            Vector s = sDesIterator.next();
            Vector y = yDesIterator.next();
            double alpha = s.dot(q) * rho;
            alphaQueue.addFirst(alpha);
            //seems no need to use "assign"
            q = q.minus(y.times(alpha));
        }

        double gamma = gamma();
        //use H_k^0 = gamma I
        Vector r = q.times(gamma);
        Iterator<Double> rhoIterator = rhoQueue.iterator();
        Iterator<Vector> sIterator = sQueue.iterator();
        Iterator<Vector> yIterator = yQueue.iterator();
        Iterator<Double> alphaIterator = alphaQueue.iterator();
        while(rhoIterator.hasNext()){
            double rho = rhoIterator.next();
            Vector s = sIterator.next();
            Vector y = yIterator.next();
            double alpha = alphaIterator.next();
            double beta = y.dot(r) * rho;
            r = r.plus(s.times(alpha - beta));
        }

        return r.times(-1);
    }

    /**
     * scaling factor
     * @return
     */
    double gamma(){
        if (sQueue.isEmpty()){
            return 1;
        }
        Vector s = sQueue.getLast();
        Vector y = yQueue.getLast();
        double denominator = y.dot(y);
        if (denominator<=0){
            return 1;
        }
        return (s.dot(y)) / (y.dot(y));
    }

    public void setHistory(double m) {
        this.m = m;
    }


}
