import sys
import os
import site
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the correct site-packages to Python path
site_packages = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Desktop', 'methane-app', 'methaneenv', 'Lib', 'site-packages')
site.addsitedir(site_packages)

print("\nPython executable:", sys.executable)
print("\nSite packages location:", site_packages)
print("\nOpenAI API Key exists:", bool(os.getenv('OPENAI_API_KEY')))

try:
    from openai import OpenAI
    print("\nOpenAI import successful!")
    client = OpenAI()  # Will use OPENAI_API_KEY from environment
    
    # Test the API connection
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10
    )
    print("OpenAI API test successful!")
    
except ImportError as e:
    print("\nOpenAI import failed:", str(e))
except Exception as e:
    print("\nAPI test failed:", str(e))