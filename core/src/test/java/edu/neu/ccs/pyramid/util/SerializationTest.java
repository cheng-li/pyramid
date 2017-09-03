package edu.neu.ccs.pyramid.util;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;

import static org.junit.Assert.*;

public class SerializationTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        String a = "sldjflsdjf";
        Serialization.serialize(a,new File(TMP,"str.ser"));
        String b = (String)Serialization.deserialize(new File(TMP, "str.ser"));
        System.out.println(b);
    }

}