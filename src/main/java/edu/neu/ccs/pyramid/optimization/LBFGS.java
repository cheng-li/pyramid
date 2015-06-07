package edu.neu.ccs.pyramid.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
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
public class LBFGS {
    private static final Logger logger = LogManager.getLogger();
    private Optimizable.ByGradientValue function;
    private BackTrackingLineSearcher lineSearcher;
    /**
     * history length;
     */
    private double m = 5;
    private LinkedList<Vector> sQueue;
    private LinkedList<Vector> yQueue;
    private LinkedList<Double> rhoQueue;
    /**
     * stop condition, relative threshold
     */
    private double epsilon = 0.01;
    private int maxIteration = 10000;
    private boolean checkConvergence =true;

    public LBFGS(Optimizable.ByGradientValue function) {
        this.function = function;
        this.lineSearcher = new BackTrackingLineSearcher(function);
        lineSearcher.setInitialStepLength(1);
        this.sQueue = new LinkedList<>();
        this.yQueue = new LinkedList<>();
        this.rhoQueue = new LinkedList<>();

    }

    public void optimize(){
        //size = 2
        LinkedList<Double> valueQueue = new LinkedList<>();
        valueQueue.add(function.getValue());
        if (logger.isDebugEnabled()){
            logger.debug("initial value = "+ valueQueue.getLast());
        }
        int iteration = 0;
        iterate();
        iteration += 1;
        valueQueue.add(function.getValue());
        if (logger.isDebugEnabled()){
            logger.debug("iteration "+iteration);
            logger.debug("value = "+valueQueue.getLast());
        }

        int convergenceTraceCounter = 0;
        while(true){

            if (checkConvergence){
                if (Math.abs(valueQueue.getFirst()-valueQueue.getLast())<epsilon*valueQueue.getFirst()){
                    convergenceTraceCounter += 1;
                } else {
                    convergenceTraceCounter =0;
                }
                if (convergenceTraceCounter == 5){
                    break;
                }
            }


            if (iteration==maxIteration){
                break;
            }
            iterate();
            iteration += 1;
            valueQueue.remove();
            valueQueue.add(function.getValue());
            if (logger.isDebugEnabled()){
                logger.debug("iteration "+iteration);
                logger.debug("value = "+valueQueue.getLast());
            }

        }

    }

    public void iterate(){
        if (logger.isDebugEnabled()){
            logger.debug("start one iteration");
        }

        Vector oldGradient = function.getGradient();
        Vector direction = findDirection();
        if (logger.isDebugEnabled()){
            logger.debug("norm of direction = "+direction.norm(2));
        }
        BackTrackingLineSearcher.MoveInfo moveInfo = lineSearcher.moveAlongDirection(direction);

        Vector s = moveInfo.getStep();
        Vector newGradient = function.getGradient();
        Vector y = newGradient.minus(oldGradient);
        double denominator = y.dot(s);

        double rho = 0;
        if (denominator>0){
            rho = 1/denominator;
        }


        if (logger.isDebugEnabled()){
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

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
    }

    public void setCheckConvergence(boolean checkConvergence) {
        this.checkConvergence = checkConvergence;
    }
}
