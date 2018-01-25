package edu.neu.ccs.pyramid.clustering.gmm;


import edu.neu.ccs.pyramid.clustering.bm.BM;
import edu.neu.ccs.pyramid.clustering.bm.BMSelector;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

public class GMMTrainerTest {
    public static void main(String[] args) throws Exception{
        test3();
    }

    private static void test2() throws Exception{
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/tmp/3gaussian.txt"));
        int dim = lines.get(0).split(" ").length;
        RealMatrix data = new Array2DRowRealMatrix(lines.size(),dim);
        for (int i=0;i<lines.size();i++){
            String[] split = lines.get(i).split(" ");
            for (int j=0;j<dim;j++){
                data.setEntry(i,j,Double.parseDouble(split[j]));
            }
        }


        GMM gmm = new GMM(dim,3, data);

        GMMTrainer trainer = new GMMTrainer(data, gmm);

        for (int i=0;i<50;i++){
            trainer.iterate();
        }

        System.out.println(gmm);
    }


    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet("/Users/chengli/Downloads/scene/train_test_split/train", DataSetType.ML_CLF_DENSE,true);
        RealMatrix data = new Array2DRowRealMatrix(dataSet.getNumDataPoints(),dataSet.getNumFeatures());
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                data.setEntry(i,j,dataSet.getRow(i).get(j));
            }
        }

        int numComponents=10;
        GMM gmm = new GMM(dataSet.getNumFeatures(),numComponents, data);

        GMMTrainer trainer = new GMMTrainer(data, gmm);


        for (int i=1;i<=5;i++){
            System.out.println("iteration = "+i);
            trainer.iterate();
            double logLikelihood = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                    .mapToDouble(j->gmm.logDensity(data.getRowVector(j))).sum();
            System.out.println("log likelihood = "+logLikelihood);
            Serialization.serialize(gmm, "/Users/chengli/tmp/gmm/model_iter_"+i);
            for (int k=0;k<gmm.getNumComponents();k++){
                FileUtils.writeStringToFile(new File("/Users/chengli/tmp/gmm/mean_iter_"+i+"_component_"+(k+1)),
                        gmm.getGaussianDistributions()[k].getMean().toString().replace("{","")
                                .replace("}","").replace(";",","));
            }
        }


    }

    private static void test3() throws Exception{
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/Downloads/fashion-mnist/features.txt"));
//        Collections.shuffle(lines);
        int dim = 784;
        int rows = 100;
        RealMatrix data = new Array2DRowRealMatrix(rows,dim);
        for (int i=0;i<rows;i++){
            String[] split = lines.get(i).split(",");
            System.out.println(Arrays.toString(split));
            for (int j=0;j<dim;j++){
                data.setEntry(i,j,Double.parseDouble(split[j])+Math.random());
            }
        }




        int numComponents = 10;
        GMM gmm = new GMM(dim,numComponents, data);

        GMMTrainer trainer = new GMMTrainer(data, gmm);


        for (int i=1;i<=5;i++){
            System.out.println("iteration = "+i);
            trainer.iterate();
            double logLikelihood = IntStream.range(0,rows).parallel()
                    .mapToDouble(j->gmm.logDensity(data.getRowVector(j))).sum();
            System.out.println("log likelihood = "+logLikelihood);
            Serialization.serialize(gmm, "/Users/chengli/tmp/gmm/model_iter_"+i);
            for (int k=0;k<gmm.getNumComponents();k++){
                FileUtils.writeStringToFile(new File("/Users/chengli/tmp/gmm/mean_iter_"+i+"_component_"+(k+1)),
                        gmm.getGaussianDistributions()[k].getMean().toString().replace("{","")
                                .replace("}","").replace(";",","));
            }
        }




    }


}