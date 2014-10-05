package edu.neu.ccs.pyramid.configuration;

import static org.junit.Assert.*;

public class ConfigTest {
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        Config config = new Config();
        config.setInt("num",1);
        config.setBoolean("read",true);
//        System.out.println(config.getInt("a"));
        System.out.println(config.getBoolean("b"));
    }

}