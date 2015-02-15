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
    private double m = 5;
    private LinkedList<Vector> sQueue;
    private LinkedList<Vector> yQueue;
    private LinkedList<Double> rhoQueue;
    /**
     * stop condition
     */
    private double epsilon = 0.1;

    public LBFGS(Optimizable.ByGradientValue function) {
        this.function = function;
        this.lineSearcher = new BackTrackingLineSearcher(function);
        lineSearcher.setInitialStepLength(1);
        this.sQueue = new LinkedList<>();
        this.yQueue = new LinkedList<>();
        this.rhoQueue = new LinkedList<>();
    }

    public void optimize(){
        LinkedList<Double> valueQueue = new LinkedList<>();
        valueQueue.add(function.getValue(function.getParameters()));
        iterate();
        valueQueue.add(function.getValue(function.getParameters()));
        while(true){
//            System.out.println("objective = "+valueQueue.getLast());
            if (Math.abs(valueQueue.getFirst()-valueQueue.getLast())<epsilon){
                break;
            }
            iterate();
            valueQueue.remove();
            valueQueue.add(function.getValue(function.getParameters()));
        }

    }

    public void iterate(){
        Vector parameters = function.getParameters();
        Vector oldGradient = function.getGradient();
        Vector direction = findDirection();
        System.out.println("doing line search");
        double stepLength = lineSearcher.findStepLength(direction);
        System.out.println("line search done");
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

    public void setHistory(double m) {
        this.m = m;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }
}
