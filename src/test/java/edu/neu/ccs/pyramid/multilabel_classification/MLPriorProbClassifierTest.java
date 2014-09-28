package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.*;


import java.util.ArrayList;

import java.util.List;


public class MLPriorProbClassifierTest {
    public static void main(String[] args) throws Exception{
        test1();
    }


    /**
     * add a fake label in spam data set, if x=spam and x_0<0.1, also label it as 2
     * @throws Exception
     */
    static void test1() throws Exception{


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet("/Users/chengli/Datasets/spam/trec_data/train.trec",
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,3);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getFeatureRow(i).getVector().get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }



        List<MultiLabel> assignments = new ArrayList<>();
        assignments.add(new MultiLabel(3).addLabel(0));
        assignments.add(new MultiLabel(3).addLabel(1));
        assignments.add(new MultiLabel(3).addLabel(1).addLabel(2));

        MLPriorProbClassifier classifier = new MLPriorProbClassifier(3,assignments);
        classifier.fit(dataSet);
        System.out.println(classifier);

    }

}