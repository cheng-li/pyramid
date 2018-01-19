package edu.neu.ccs.pyramid.clustering.gmm;


import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.File;
import java.util.List;

public class GMMTrainerTest {
    public static void main(String[] args) throws Exception{
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



}