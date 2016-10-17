package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.multilabel_classification.AccScorer;
import edu.neu.ccs.pyramid.multilabel_classification.FScorer;
import edu.neu.ccs.pyramid.multilabel_classification.MLScorer;
import edu.neu.ccs.pyramid.simulation.MultiLabelSynthesizer;
import junit.framework.TestCase;

/**
 * Created by chengli on 10/17/16.
 */
public class NoiseOptimizerTest{
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        MultiLabelClfDataSet train = MultiLabelSynthesizer.crfArgmaxDrop();
        MultiLabelClfDataSet test = MultiLabelSynthesizer.crfArgmax();
        CMLCRF cmlcrf = new CMLCRF(train);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(0,0);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(1,0);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(0,-10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(1,-10);


        MLScorer accScorer = new AccScorer();

        SubsetAccPredictor plugInAcc = new SubsetAccPredictor(cmlcrf);


        InstanceF1Predictor plugInF1 = new InstanceF1Predictor(cmlcrf);

        System.out.println(cmlcrf);
        System.out.println("training performance acc");
        System.out.println(new MLMeasures(cmlcrf, train));
        System.out.println("test performance acc");
        System.out.println(new MLMeasures(cmlcrf, test));
        System.out.println("training performance f1");
        System.out.println(new MLMeasures(plugInF1, train));
        System.out.println("test performance f1");
        System.out.println(new MLMeasures(plugInF1, test));





        LogRiskOptimizer accOptimizer = new LogRiskOptimizer(train, accScorer, cmlcrf, 1, false, false, 1, 1);
        accOptimizer.iterate();
        System.out.println("after ML estimation");
        System.out.println("training with Acc predictor");
        System.out.println(new MLMeasures(plugInAcc,train));

        System.out.println("training with F1 predictor");
        System.out.println(new MLMeasures(plugInF1,train));
        System.out.println("test with Acc predictor");
        System.out.println(new MLMeasures(plugInAcc,test));
        System.out.println("test with F1 predictor");
        System.out.println(new MLMeasures(plugInF1,test));

        System.out.println(cmlcrf);


        NoiseOptimizer noiseOptimizer = new NoiseOptimizer(train, cmlcrf, 1);

        while (!noiseOptimizer.getTerminator().shouldTerminate()) {
            System.out.println("------------");
            noiseOptimizer.iterate();
            System.out.println(noiseOptimizer.objectiveDetail());
            System.out.println("training performance acc");
            System.out.println(new MLMeasures(cmlcrf, train));
            System.out.println("test performance acc");
            System.out.println(new MLMeasures(cmlcrf, test));
            System.out.println("training performance f1");
            System.out.println(new MLMeasures(plugInF1, train));
            System.out.println("test performance f1");
            System.out.println(new MLMeasures(plugInF1, test));
            System.out.println(cmlcrf);
        }
    }

}