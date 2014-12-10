package edu.neu.ccs.pyramid.optimization;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Iterator;
import java.util.LinkedList;


/**
 * Numerical Optimization, Second Edition, Jorge Nocedal Stephen J. Wright
 * Algorithm 7.4 and 7.5
 * Created by chengli on 12/9/14.
 */
public class LBFGS {
    private Optimizable.ByGradientValue function;
    private BackTrackingLineSearcher lineSearcher;
    /**
     * history length;
     */
    private double m;
    private LinkedList<Vector> sQueue;
    private LinkedList<Vector> yQueue;
    private LinkedList<Double> rhoQueue;

    public LBFGS(Optimizable.ByGradientValue function,
                                    int historyLength) {
        this.function = function;
        this.lineSearcher = new BackTrackingLineSearcher(function);
        lineSearcher.setInitialStepLength(1);
        this.m = historyLength;
        this.sQueue = new LinkedList<>();
        this.yQueue = new LinkedList<>();
        this.rhoQueue = new LinkedList<>();
    }

    public void update(){
        Vector parameters = function.getParameters();
        Vector oldGradient = function.getGradient();
        Vector direction = findDirection();
        double stepLength = lineSearcher.findStepLength(direction);
        System.out.println("stepLength="+stepLength);
        Vector s = direction.times(stepLength);
        Vector updatedParams = parameters.plus(s);
        parameters.assign(updatedParams);
        function.refresh();
        Vector newGradient = function.getGradient();
        Vector y = newGradient.minus(oldGradient);
        double rho = 1/(y.dot(s));
        sQueue.add(s);
        yQueue.add(y);
        rhoQueue.add(rho);
        if (sQueue.size()>m){
            sQueue.remove();
            yQueue.remove();
            rhoQueue.remove();
        }
    }

    Vector findDirection(){
        Vector g = function.getGradient();
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
        return (s.dot(y)) / (y.dot(y));
    }

    public void setHistoryLength(double m) {
        this.m = m;
    }

}
