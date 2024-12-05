@echo off
mkdir lib 2>nul
cd lib

:: Download required JARs
curl -O https://repo1.maven.org/maven2/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar
curl -O https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/8.11.2/lucene-core-8.11.2.jar
curl -O https://repo1.maven.org/maven2/org/apache/lucene/lucene-analyzers-common/8.11.2/lucene-analyzers-common-8.11.2.jar
curl -O https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-stable/3.8.6/weka-stable-3.8.6.jar

echo Dependencies downloaded to lib folder
