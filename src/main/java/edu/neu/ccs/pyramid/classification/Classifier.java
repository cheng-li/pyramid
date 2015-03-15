package edu.neu.ccs.pyramid.classification;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.FeatureList;
import org.apache.mahout.math.Vector;


import java.io.*;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/13/14.
 */
public interface Classifier extends Serializable{
    int predict(Vector vector);

    int  getNumClasses();

    default int[] predict(ClfDataSet dataSet){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().
                map(i -> predict(dataSet.getRow(i))).toArray();
    }

    default void serialize(File file) throws Exception{
        File parent = file.getParentFile();
        if (!parent.exists()){
            parent.mkdirs();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(this);
        }
    }

    default void serialize(String file) throws Exception{
        serialize(new File(file));
    }

    FeatureList getFeatureList();

    LabelTranslator getLabelTranslator();


}