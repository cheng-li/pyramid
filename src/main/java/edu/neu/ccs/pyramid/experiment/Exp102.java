package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.util.Serialization;

/**
 * Created by chengli on 5/6/15.
 */
public class Exp102 {
    public static void main(String[] args) throws Exception{
        int[] list = {1,2,3,5,6,8,9,10,11,12,13,14,15};
        String folder = "/huge1/people/chengli/projects/pyramid/archives/exp99/amazon_baby/";
        for (int i:list){
            String path = folder+i+"/fold_1/";
            LogisticRegression ridge = (LogisticRegression)Serialization.deserialize(path+"ridge.ser");
            LogisticRegression lasso = (LogisticRegression)Serialization.deserialize(path+"lasso.ser");
            LogisticRegression elastic = (LogisticRegression)Serialization.deserialize(path+"elastic.ser");
            System.out.println(""+i);
            System.out.print(LogisticRegressionInspector.numOfUsedFeaturesCombined(ridge));
            System.out.print(",");
            System.out.print(LogisticRegressionInspector.numOfUsedFeaturesCombined(lasso));
            System.out.print(",");
            System.out.println(LogisticRegressionInspector.numOfUsedFeaturesCombined(elastic));
        }
    }
}
