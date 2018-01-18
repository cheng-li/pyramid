package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.clustering.gmm.GMM;
import edu.neu.ccs.pyramid.clustering.gmm.GMMTrainer;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class GMMDemo {
    public static void main(String[] args) throws Exception{
        List<String> lines = FileUtils.readLines(new File("/Users/chengli/Dropbox/Shared/CS6220DM/2_cluster_EM_mixt/HW2/mnist_features.txt"));
        int dim = lines.get(0).split(" ").length;
        int rows = 1000;
        RealMatrix data = new Array2DRowRealMatrix(rows,dim);
        for (int i=0;i<rows;i++){
            String[] split = lines.get(i).split(" ");
            for (int j=0;j<dim;j++){
                data.setEntry(i,j,Double.parseDouble(split[j])+Math.random());
            }
        }


        GMM gmm = new GMM(dim,5);

        GMMTrainer trainer = new GMMTrainer(data, gmm);

        for (int i=1;i<=3;i++){
            System.out.println("iteration = "+i);
            trainer.iterate();
            double logLikelihood = IntStream.range(0,rows).parallel()
                    .mapToDouble(j->gmm.logDensity(data.getRowVector(j))).sum();
            System.out.println("log likelihood = "+logLikelihood);
            Serialization.serialize(gmm, "/Users/chengli/tmp/gmm/model_"+i);
        }

        System.out.println(gmm);


//        GMM gmm = (GMM) Serialization.deserialize("/Users/chengli/tmp/gmm/model_1");
        System.out.println(gmm.getGaussianDistributions()[0].getInverseCovariance());
        System.out.println(Arrays.toString(gmm.posteriors(data.getRowVector(0))));
        System.out.println(gmm.getGaussianDistributions()[0].logDensity(data.getRowVector(0)));

        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/gmm/modeltext"), gmm.toString());
        System.out.println(gmm);
    }
}
