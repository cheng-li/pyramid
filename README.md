Pyramid Machine Learning Library

## **Requirements**
If you just want to use pyramid as a command line tool, all you need is [Java 8](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html).

If you are a Java developer and wish to call pyramid Java APIs, you will also need [Maven](https://maven.apache.org/).

## **Setup**
Pyramid doesn't require any installation effort. You can simply download a pre-compiled package from the [project release page](https://github.com/cheng-li/pyramid/releases). Decompress the file, move into the created folder and type 

`./pyramid welcome config/welcome.properties`

A welcome message will show up on your screen.
## **Command Line Usage**
All commands are in the following format:

`./pyramid <app_name> <properties_file>`

The `<app_name>` is case-insensitive.

The `<properties_file>` can be specified by either an absolute or a relative path.

Example: 

`./pyramid welcome config/welcome.properties`

or

`./pyramid app1 config/app1.properties`
## **Building from Source**
Pyramid uses [Maven](https://maven.apache.org/) for its build system.

To compile and package the project, simply run the `mvn clean package -DskipTests` command in the cloned directory. The compressed package will be created under the target/releases directory. Just decompress either the .zip file or the .tar.gz file and you are done.
