# Java Performance Optimizer Setup

## Project Structure
- Source Code: `src/java/`
- Package: `com.albert.trading.optimizer`

## Build and Configuration
1. Ensure you have Maven installed
2. Run `mvn clean install` to download dependencies
3. Configure your IDE:
   - Import as Maven project
   - Set source directory to `src/java/`
   - Ensure Java 11+ is used

## Troubleshooting
- If build path issues persist, manually add project as a Java source folder in your IDE
- Verify that `JAVA_HOME` is correctly set
- Check that Maven can resolve project dependencies

## Dependencies
- Java 11+
- Maven 3.6+
- Additional libraries managed via `pom.xml`
