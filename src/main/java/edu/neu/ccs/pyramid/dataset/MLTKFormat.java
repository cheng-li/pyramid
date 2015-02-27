//package edu.neu.ccs.pyramid.dataset;
//
//import edu.neu.ccs.pyramid.util.Pair;
//import mltk.core.*;
//import org.apache.mahout.math.Vector;
//
//import java.util.ArrayList;
//import java.util.Comparator;
//import java.util.List;
//import java.util.stream.Collectors;
//
///** @deprecated
// * Created by chengli on 11/18/14.
// */
//public class MLTKFormat {
//    public static Instances toInstances(ClfDataSet dataSet){
//        List<Attribute> attributes = new ArrayList<>();
//        for (int j=0;j<dataSet.getNumFeatures();j++){
//            String name = dataSet.getFeatureSetting(j).getFeatureName();
//            attributes.add(new NumericalAttribute(name,j));
//        }
//        String[] states = new String[dataSet.getNumClasses()];
//        for (int k=0;k<states.length;k++){
//            states[k]=""+k;
//        }
//
//        Attribute targetAttribute = new NominalAttribute("target",states);
//
//        Instances instances = new Instances(attributes, targetAttribute);
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            instances.add(toInstance(dataSet,i));
//        }
//        return instances;
//    }
//
//    /**
//     * translate to a sparse instance
//     * using dense instance seems to make the learning slow
//     * @param dataSet
//     * @param dataPointIndex
//     * @return
//     */
//    static Instance toInstance(ClfDataSet dataSet, int dataPointIndex){
//        Vector vector = dataSet.getRow(dataPointIndex);
//
//        List<Pair<Integer,Double>> pairs = new ArrayList<>();
//        for (Vector.Element element:vector.nonZeroes()){
//            Pair<Integer,Double> pair = new Pair<>(element.index(),element.get());
//            pairs.add(pair);
//        }
//        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getFirst);
//        List<Pair<Integer,Double>> sorted = pairs.stream().sorted(comparator)
//                .collect(Collectors.toList());
//
//        int[] indices = new int[sorted.size()];
//        double[] values = new double[sorted.size()];
//        for (int i=0;i<sorted.size();i++){
//            Pair<Integer,Double> pair = sorted.get(i);
//            indices[i] = pair.getFirst();
//            values[i] = pair.getSecond();
//        }
//
//        double label = dataSet.getLabels()[dataPointIndex];
//        return new Instance(indices,values,label);
//    }
//}
