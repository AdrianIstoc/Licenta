from App.application import Application
from SentinelHub.sh_connection import SentinelHubConnection

def main():
    connection = SentinelHubConnection()
    config = connection.get_config()

    app = Application(config=config)
    app.mainloop()

if __name__ == "__main__":
    main()