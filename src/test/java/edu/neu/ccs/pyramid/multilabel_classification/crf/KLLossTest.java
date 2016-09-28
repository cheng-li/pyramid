package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import junit.framework.TestCase;

import java.io.File;
import java.util.List;

/**
 * Created by chengli on 9/28/16.
 */
public class KLLossTest extends TestCase {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }


    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "spam/trec_data/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "spam/trec_data/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);

        CMLCRF cmlcrf = new CMLCRF(dataSet);
        List<MultiLabel> support = cmlcrf.getSupportCombinations();
        double[][] targetDistribution = new double[dataSet.getNumDataPoints()][cmlcrf.getSupportCombinations().size()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int c=0;c<support.size();c++){
                MultiLabel multiLabel = dataSet.getMultiLabels()[i];
                if (support.get(c).equals(multiLabel)){
                    targetDistribution[i][c] = 1;
                }
            }
        }


        System.out.println("start");
        KLLoss klLoss = new KLLoss(cmlcrf, dataSet,targetDistribution, 1);
        cmlcrf.setConsiderPair(true);


        MultiLabel[] predTrain;
        MultiLabel[] predTest;

        LBFGS optimizer = new LBFGS(klLoss);
        for (int i=0; i<200; i++) {

//            System.out.print("Obj: " + optimizer.getTerminator().getLastValue());
            System.out.println("iter: "+ i);
            optimizer.iterate();
            System.out.println(klLoss.getValue());
            predTrain = cmlcrf.predict(dataSet);
            predTest = cmlcrf.predict(testSet);
            System.out.print("\tTrain acc: " + Accuracy.accuracy(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTrain overlap " + Overlap.overlap(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTest acc: " + Accuracy.accuracy(testSet.getMultiLabels(), predTest));
            System.out.println("\tTest overlap " + Overlap.overlap(testSet.getMultiLabels(), predTest));

        }

    }
}