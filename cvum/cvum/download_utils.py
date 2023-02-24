import requests

def download_file_from_google_drive(file_id:str, destination_folder:str):
    ''' function from: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive#:~:text=I%20wrote%20a%20Python%20snippet%20that%20downloads%20a%20file%20from%20Google%20Drive%2C%20given%20a%20shareable%20link.%20It%20works%2C%20as%20of%20August%202017.'''
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination_folder):
        CHUNK_SIZE = 32768

        with open(destination_folder, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination_folder)    