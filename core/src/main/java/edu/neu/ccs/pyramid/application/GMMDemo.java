package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.clustering.bm.BMSelector;
import edu.neu.ccs.pyramid.clustering.bm.BMTrainer;
import edu.neu.ccs.pyramid.clustering.gmm.GMM;
import edu.neu.ccs.pyramid.clustering.gmm.GMMTrainer;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
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

public class GMMDemo {
    public static void main(String[] args) throws Exception{
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/Dropbox/Shared/CS6220DM/2_cluster_EM_mixt/HW2/mnist_features.txt"));
        Collections.shuffle(lines);
        int dim = lines.get(0).split(" ").length;
        int rows = 100;
        RealMatrix data = new Array2DRowRealMatrix(rows,dim);
        for (int i=0;i<rows;i++){
            String[] split = lines.get(i).split(" ");
            for (int j=0;j<dim;j++){
                data.setEntry(i,j,Double.parseDouble(split[j])+Math.random());
            }
        }

        double[] mins = new double[data.getColumnDimension()];
        double[] maxs = new double[data.getColumnDimension()];
        double[] vars = new double[data.getColumnDimension()];
        for (int j=0;j<data.getColumnDimension();j++){
            RealVector column = data.getColumnVector(j);
            mins[j] = column.getMinValue();
            maxs[j] = column.getMaxValue();
            DescriptiveStatistics stats = new DescriptiveStatistics(column.toArray());
            vars[j] = stats.getVariance();
        }

        DataSet dataSet = DataSetBuilder.getBuilder()
                .numDataPoints(rows)
                .numFeatures(data.getColumnDimension())
                .build();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                if (data.getEntry(i,j)>255.0/2){
                    dataSet.setFeatureValue(i,j,1);
                } else {
                    dataSet.setFeatureValue(i,j,0);
                }

            }
        }

        int numComponents = 10;
        BMTrainer bmTrainer = BMSelector.selectTrainer(dataSet,numComponents,5);


        GMM gmm = new GMM(dim,numComponents, data);

        GMMTrainer trainer = new GMMTrainer(data, gmm);

        trainer.setGammas(bmTrainer.getGammas());
        trainer.mStep();

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

        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/gmm/modeltext"), gmm.toString());

//        GMM gmm = (GMM) Serialization.deserialize("/Users/chengli/tmp/gmm/model_3");




    }
}
