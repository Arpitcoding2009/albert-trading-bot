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

## Environment Setup

1. Copy `.env.template` to `.env`:
```bash
cp .env.template .env
```

2. Update the `.env` file with your API keys:
- OpenAI API key for GPT-4 access
- Anthropic API key for Claude access (optional)
- Exchange API keys (Binance, Coinbase, Kraken)

3. Configure trading parameters in `.env`:
- Risk tolerance
- Maximum position size
- Stop loss percentage
- Model settings (temperature, max tokens)

**Important**: Never commit your `.env` file to version control. It's already added to `.gitignore`.

## Troubleshooting
- If build path issues persist, manually add project as a Java source folder in your IDE
- Verify that `JAVA_HOME` is correctly set
- Check that Maven can resolve project dependencies

## Dependencies
- Java 11+
- Maven 3.6+
- Additional libraries managed via `pom.xml`
