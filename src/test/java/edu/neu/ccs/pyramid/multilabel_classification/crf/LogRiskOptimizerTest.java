package edu.neu.ccs.pyramid.multilabel_classification.crf;


import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.multilabel_classification.AccScorer;
import edu.neu.ccs.pyramid.multilabel_classification.FScorer;
import edu.neu.ccs.pyramid.multilabel_classification.MLScorer;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.simulation.MultiLabelSynthesizer;

/**
 * Created by chengli on 10/16/16.
 */
public class LogRiskOptimizerTest {
    public static void main(String[] args) {
        test4();
    }

    private static void test1(){
        MultiLabelClfDataSet train = MultiLabelSynthesizer.independentNoise();
        MultiLabelClfDataSet test = MultiLabelSynthesizer.independent();
        CMLCRF cmlcrf = new CMLCRF(train);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(0,0);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(1,1);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(0,1);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(1,1);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(0,1);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(1,0);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(0,1);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(1,-1);



        MLScorer fScorer = new FScorer();
        LogRiskOptimizer fOptimizer = new LogRiskOptimizer(train, fScorer, cmlcrf, 1, false, false, 1, 1);
        InstanceF1Predictor plugInF1 = new InstanceF1Predictor(cmlcrf);

        System.out.println(cmlcrf);
        System.out.println("initial loss = "+fOptimizer.objective());
        System.out.println("training performance acc");
        System.out.println(new MLMeasures(cmlcrf, train));
        System.out.println("test performance acc");
        System.out.println(new MLMeasures(cmlcrf, test));
        System.out.println("training performance f1");
        System.out.println(new MLMeasures(plugInF1, train));
        System.out.println("test performance f1");
        System.out.println(new MLMeasures(plugInF1, test));





        while (!fOptimizer.getTerminator().shouldTerminate()) {
            System.out.println("------------");
            fOptimizer.iterate();
            System.out.println(fOptimizer.getTerminator().getLastValue());
            System.out.println("training performance acc");
            System.out.println(new MLMeasures(cmlcrf, train));
            System.out.println("test performance acc");
            System.out.println(new MLMeasures(cmlcrf, test));
            System.out.println("training performance f1");
            System.out.println(new MLMeasures(plugInF1, train));
            System.out.println("test performance f1");
            System.out.println(new MLMeasures(plugInF1, test));
        }
        System.out.println(cmlcrf);
    }

    private static void test2(){
        MultiLabelClfDataSet train = MultiLabelSynthesizer.independentNoise();
        MultiLabelClfDataSet test = MultiLabelSynthesizer.independent();
        CMLCRF cmlcrf = new CMLCRF(train);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(0,0);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(1,1);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(0,1);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(1,1);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(0,1);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(1,0);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(0,1);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(1,-1);

        InstanceF1Predictor plugInF1 = new InstanceF1Predictor(cmlcrf);

        for (int i=0;i<test.getNumDataPoints();i++){
            System.out.println("=============");
            System.out.println(i);
            System.out.println(plugInF1.showPredictBySupport(test.getRow(i),test.getMultiLabels()[i]));
        }
    }

    private static void test3(){
        MultiLabelClfDataSet train = MultiLabelSynthesizer.crfSample();
        MultiLabelClfDataSet test = MultiLabelSynthesizer.crfSample();
        CMLCRF cmlcrf = new CMLCRF(train);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(0,0);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(1,0);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(0,-10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(1,-10);



        MLScorer fScorer = new FScorer();
        MLScorer accScorer = new AccScorer();

        SubsetAccPredictor plugInAcc = new SubsetAccPredictor(cmlcrf);

        LogRiskOptimizer fOptimizer = new LogRiskOptimizer(train, fScorer, cmlcrf, 1, false, false, 1, 1);
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

        System.out.println(fOptimizer.objectiveDetail());

        while (!fOptimizer.getTerminator().shouldTerminate()) {
            System.out.println("------------");
            fOptimizer.iterate();
            System.out.println(fOptimizer.objectiveDetail());
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

    private static void test4(){
        MultiLabelClfDataSet train = MultiLabelSynthesizer.crfArgmaxHide();
        MultiLabelClfDataSet test = MultiLabelSynthesizer.crfArgmaxHide();
        CMLCRF cmlcrf = new CMLCRF(train);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(0,0);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(1,0);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(0,-10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(1,-10);



        MLScorer fScorer = new FScorer();
        MLScorer accScorer = new AccScorer();

        SubsetAccPredictor plugInAcc = new SubsetAccPredictor(cmlcrf);

        LogRiskOptimizer fOptimizer = new LogRiskOptimizer(train, fScorer, cmlcrf, 1, false, false, 1, 1);
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

        System.out.println(fOptimizer.objectiveDetail());



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

        System.out.println(fOptimizer.objectiveDetail());

        while (!fOptimizer.getTerminator().shouldTerminate()) {
            System.out.println("------------");
            fOptimizer.iterate();
            System.out.println(fOptimizer.objectiveDetail());
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