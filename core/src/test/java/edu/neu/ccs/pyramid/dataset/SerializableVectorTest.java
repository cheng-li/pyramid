package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.util.Serialization;
import junit.framework.TestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.File;

/**
 * Created by chengli on 11/27/16.
 */
public class SerializableVectorTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        Vector vector = new SequentialAccessSparseVector(10);
        vector.set(0,2);
        vector.set(5,8);
        SerializableVector serializableVector = new SerializableVector(vector);
        Serialization.serialize(serializableVector, new File(TMP,"v.ser"));
        Vector loaded = ((SerializableVector)Serialization.deserialize(new File(TMP,"v.ser"))).getVector();
        System.out.println(loaded.size());
        System.out.println(loaded.getClass().getName());
        System.out.println(loaded);
    }

}