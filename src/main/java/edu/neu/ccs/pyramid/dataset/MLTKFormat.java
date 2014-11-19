package edu.neu.ccs.pyramid.dataset;

import mltk.core.*;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 11/18/14.
 */
public class MLTKFormat {
    public static Instances toInstances(ClfDataSet dataSet){
        List<Attribute> attributes = new ArrayList<>();
        for (int j=0;j<dataSet.getNumFeatures();j++){
            String name = dataSet.getFeatureSetting(j).getFeatureName();
            attributes.add(new NumericalAttribute(name,j));
        }
        String[] states = new String[dataSet.getNumClasses()];
        for (int k=0;k<states.length;k++){
            states[k]=""+k;
        }

        Attribute targetAttribute = new NominalAttribute("target",states);

        Instances instances = new Instances(attributes, targetAttribute);
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            instances.add(toInstance(dataSet,i));
        }
        return instances;
    }

    static Instance toInstance(ClfDataSet dataSet, int dataPointIndex){
        double[] features = new double[dataSet.getNumFeatures()];
        Vector vector = dataSet.getRow(dataPointIndex);
        for (Vector.Element element: vector.nonZeroes()){
            int index = element.index();
            double value = element.get();
            features[index] = value;
        }
        double label = dataSet.getLabels()[dataPointIndex];
        return new Instance(features,label);
    }
}
