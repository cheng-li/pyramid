package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import org.apache.mahout.math.Vector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Created by chengli on 2/9/15.
 */
public class Trec2Matlab {
    public static void main(String[] args) throws Exception{
        Config config = new Config(args[0]);
        File trecFile = new File(config.getString("input.trecFile"));
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(trecFile, DataSetType.CLF_SPARSE,false);
        File matlabFile = new File(config.getString("output.matlabFile"));
        matlabFile.getParentFile().mkdirs();
        try(BufferedWriter bw = new BufferedWriter(new FileWriter(matlabFile))
        ){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                Vector vector = dataSet.getRow(i);
                for (Vector.Element element: vector.nonZeroes()){
                    int j= element.index();
                    double value = element.get();
                    bw.write(""+(i+1));
                    bw.write("\t");
                    bw.write(""+(j+1));
                    bw.write("\t");
                    bw.write(""+value);
                    bw.newLine();
                }
            }
        }
    }
}
