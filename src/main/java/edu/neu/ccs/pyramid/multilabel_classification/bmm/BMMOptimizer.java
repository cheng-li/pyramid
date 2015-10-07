package edu.neu.ccs.pyramid.multilabel_classification.bmm;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.*;

/**
 * Created by chengli on 10/7/15.
 */
public class BMMOptimizer {
    private BMMClassifier bmmClassifier;
    private MultiLabelClfDataSet dataSet;
    private Terminator terminator;
    double[][] gammas;

    public void optimize(){
        while (true){
            iterate();
            if (terminator.shouldTerminate()){
                break;
            }
        }
    }


    private void iterate(){
        eStep();
        mStep();
        this.terminator.add(getObjective());
    }

    private void eStep(){

    }


    private void mStep(){

    }

    private double getObjective(){
        return 0;
    }
}
