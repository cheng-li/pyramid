package edu.neu.ccs.pyramid.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.LinkedList;

/**
 * Created by chengli on 12/7/14.
 */
public class GradientDescent {
    private static final Logger logger = LogManager.getLogger();
    private Optimizable.ByGradientValue function;
    private BackTrackingLineSearcher lineSearcher;
    /**
     * stop condition, relative threshold
     */
    private double epsilon = 0.01;
    private int maxIteration = 10000;
    private boolean checkConvergence =true;
    private boolean terminate = false;


    public GradientDescent(Optimizable.ByGradientValue function,
                           double initialStepLength) {
        this.function = function;
        this.lineSearcher = new BackTrackingLineSearcher(function);
        lineSearcher.setInitialStepLength(initialStepLength);
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
                    terminate = true;
                }
            }


            if (iteration==maxIteration){
                terminate = true;
            }

            if (terminate){
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
        Vector gradient = function.getGradient();
        Vector direction = gradient.times(-1);
        lineSearcher.moveAlongDirection(direction);

    }

    public void setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
    }

    public void setCheckConvergence(boolean checkConvergence) {
        this.checkConvergence = checkConvergence;
    }
}
