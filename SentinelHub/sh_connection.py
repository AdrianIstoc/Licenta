from sentinelhub import SHConfig, SentinelHubCatalog
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
BASE_URL = os.getenv("BASE_URL")
TOKEN_URL = os.getenv("TOKEN_URL")

class SentinelHubConnection:

    def __init__(self):

        if not CLIENT_ID:
            raise ValueError("CLIENT_ID is missing")
        if not CLIENT_SECRET:
            raise ValueError("CLIENT_SECRET is missing")

        self.config = SHConfig()

        self.config.sh_client_id = CLIENT_ID
        self.config.sh_client_secret = CLIENT_SECRET
        self.config.sh_base_url = BASE_URL
        self.config.sh_token_url = TOKEN_URL


    def get_config(self):

        return self.config
        

    def get_catalog(self):
        
        return SentinelHubCatalog(config=self.config)
    

    def test_connection(self):

        try:

            catalog = self.get_catalog()

            collections = catalog.get_collections()

            print(f"Found {len(collections)} collections.")

            return True
        
        except Exception as e:

            print(e)

            return False
        