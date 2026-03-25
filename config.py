import os
import configparser


class Config:
    """
    Configuration management class for handling application settings.

    This class provides a wrapper around Python's configparser to read and access
    configuration values from INI files with type conversion methods and
    specialized initialization methods for HuggingFace environment setup.
    """

    def __init__(self, filename="config.ini"):
        """
        Initialize the configuration manager.

        Args:
            filename: Path to the configuration INI file (default: "config.ini")

        Raises:
            FileNotFoundError: If the specified configuration file does not exist
        """
        # Initialize ConfigParser instance
        self.cfg = configparser.ConfigParser()

        # Check if configuration file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Configuration file not found: {filename}")

        # Read and parse the configuration file with UTF-8 encoding
        self.cfg.read(filename, encoding="utf-8")

    def get(self, section, key, fallback=None):
        """
        Get a string value from the configuration.

        Args:
            section: Configuration section name
            key: Configuration key within the section
            fallback: Default value to return if key is not found

        Returns:
            String value from configuration or fallback
        """
        return self.cfg.get(section, key, fallback=fallback)

    def get_int(self, section, key, fallback=None):
        """
        Get an integer value from the configuration.

        Args:
            section: Configuration section name
            key: Configuration key within the section
            fallback: Default value to return if key is not found

        Returns:
            Integer value from configuration or fallback
        """
        return self.cfg.getint(section, key, fallback=fallback)

    def get_float(self, section, key, fallback=None):
        """
        Get a float value from the configuration.

        Args:
            section: Configuration section name
            key: Configuration key within the section
            fallback: Default value to return if key is not found

        Returns:
            Float value from configuration or fallback
        """
        return self.cfg.getfloat(section, key, fallback=fallback)

    def get_bool(self, section, key, fallback=None):
        """
        Get a boolean value from the configuration.

        Args:
            section: Configuration section name
            key: Configuration key within the section
            fallback: Default value to return if key is not found

        Returns:
            Boolean value from configuration or fallback
        """
        return self.cfg.getboolean(section, key, fallback=fallback)

    def init_HuggingFace(self):
        """
        Initialize HuggingFace environment variables.

        This method sets up cache directories and endpoint configurations
        for HuggingFace libraries (transformers, datasets, etc.) to use
        a centralized cache location and a mirror endpoint for improved
        accessibility in specific regions.
        """
        import os

        # Set cache directories for HuggingFace libraries
        os.environ["HF_HOME"] = "/data/hf_cache"  # Main HuggingFace cache
        os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"  # Transformers library cache
        os.environ["HF_DATASETS_CACHE"] = "/data/hf_cache/datasets"  # Datasets library cache

        # Set mirror endpoint for HuggingFace (useful in regions with restricted access)
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # Print initialization confirmation
        print("HuggingFace initialization completed---\n")


# Create a globally accessible configuration instance
# This follows the singleton pattern to ensure consistent configuration access throughout the application
cfg = Config()