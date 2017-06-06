package edu.neu.ccs.pyramid.core.multilabel_classification;

import edu.neu.ccs.pyramid.core.dataset.*;


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
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(3).build();
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getRow(i).get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }



        MLPriorProbClassifier classifier = new MLPriorProbClassifier(3);
        classifier.fit(dataSet);
        System.out.println(classifier);

    }

}