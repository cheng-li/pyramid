package edu.neu.ccs.pyramid.core.configuration;

import java.io.File;

public class ConfigTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test3();
    }

    private static void test1(){
        Config config = new Config();
        config.setInt("num",1);
        config.setBoolean("read",true);
//        System.out.println(config.getInt("a"));
        System.out.println(config.getBoolean("b"));
    }

    private static void test2() throws Exception{
        for (int i=0;i<100000;i++){
            Config config = new Config();
            config.setInt("num",1);
            config.setBoolean("read",true);
            config.store(new File(TMP,""+i));
        }

    }

    private static void test3() throws Exception{
        for (int i=0;i<100000;i++){
            Config config = new Config(new File(TMP,""+i));
//            System.out.println(config);
        }

    }

}