package edu.neu.ccs.pyramid.util;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Commander {
    /**
     * run system command and collect output
     * @param command
     * @return
     * @throws Exception
     */
    public static List<String> runCommand(String command) throws Exception{
        String[] fullCommand = {"/bin/sh", "-c", command};
        List<String> out = new ArrayList<>();
        Runtime r = Runtime.getRuntime();
        Process p = r.exec(fullCommand);
        p.waitFor();
        BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
        String line = "";

        while ((line = b.readLine()) != null) {
            out.add(line);
        }

        b.close();
        return out;
    }
}
