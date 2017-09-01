package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Serialization;
import junit.framework.TestCase;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 4/2/17.
 */
public class ShortCircuitPosteriorTest {
    public static void main(String[] args) throws Exception{
//        System.out.println((0.0+10)/(30.0+10000));
        double[] s = {-40,-40,-20, 0};
        System.out.println(Arrays.toString(MathUtil.softmax(s)));
//        System.out.println(MathUtil.logSoftmax(s)[0]);
//        System.out.println(Math.exp(-20));

        //        MultiLabelClfDataSet train = TRECFormat.loadMultiLabelClfDataSet("/Users/chengli/tmp/mlc_data_pyramid/rcv1subset_topics_1/train_test_split/train", DataSetType.ML_CLF_SEQ_SPARSE,true);
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet("/Users/chengli/tmp/mlc_data_pyramid/rcv1subset_topics_1/train_test_split/test", DataSetType.ML_CLF_SEQ_SPARSE,true);

//        boolean[] check = new boolean[train.getNumClasses()];
//        for (int i=0;i<train.getNumDataPoints();i++){
//            MultiLabel multiLabel = train.getMultiLabels()[i];
//            for (int l:multiLabel.getMatchedLabels()){
//                check[l]=true;
//            }
//        }

//        System.out.println(Arrays.toString(check));

        int dataIndex = 190;

        CBM cbm = (CBM) Serialization.deserialize("/Users/chengli/tmp/model");
        BMDistribution distribution = cbm.computeBM(test.getRow(dataIndex));
        System.out.println("pi");
        System.out.println(Arrays.toString(cbm.getMultiClassClassifier().predictClassProbs(test.getRow(dataIndex))));
        System.out.println("posterior");
        System.out.println(Arrays.toString(distribution.posteriorMembership(test.getMultiLabels()[dataIndex])));
        System.out.println("approximate posterior = ");
        System.out.println(Arrays.toString(new ShortCircuitPosterior(cbm, test.getRow(dataIndex), test.getMultiLabels()[dataIndex]).posteriorMembership()));
        System.out.println(Arrays.toString(distribution.getLogClassProbs()));

        for (int k=0;k<cbm.getNumComponents();k++){
            System.out.println("k="+k);
            System.out.println(cbm.getMultiClassClassifier().predictLogClassProbs(test.getRow(dataIndex))[k]);
            System.out.println(distribution.logYGivenComponentByDefault(test.getMultiLabels()[dataIndex], k));
            System.out.println(distribution.posteriorMembership(test.getMultiLabels()[dataIndex])[k]);

        }



        double[][][] logClassProbs = distribution.getLogClassProbs();
        for (int l=0;l<test.getNumClasses();l++){
            final int label = l;
            double max = IntStream.range(0, cbm.getNumComponents()).mapToDouble(k->logClassProbs[k][label][1]).max().getAsDouble();
            System.out.println("label "+l);
            System.out.println("max = "+max);
        }


        System.out.println(distribution.logProbability(test.getMultiLabels()[dataIndex]));
        for (int k=0;k<cbm.getNumComponents();k++){
            System.out.println(distribution.logYGivenComponentByDefault(test.getMultiLabels()[dataIndex],k));
        }

//        System.out.println(cbm.predictLogAssignmentProb(test.getRow(dataIndex),test.getMultiLabels()[dataIndex]));
    }

}