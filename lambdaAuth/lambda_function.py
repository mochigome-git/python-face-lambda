import boto3
import json
import os
from base64 import b64decode

# Decrypt the environment variable
ENCRYPTED = os.environ['ANON_KEY']

# Version encrypting env variable 
# Ensure the decrypted value is stored globally to prevent repeated decryption
DECRYPTED = boto3.client('kms').decrypt(
    CiphertextBlob=b64decode(ENCRYPTED),
    EncryptionContext={'LambdaFunctionName': os.environ['AWS_LAMBDA_FUNCTION_NAME']}
)['Plaintext'].decode('utf-8')

def lambda_handler(event, context):
    # Ensure 'headers' key exists in the event
    if 'headers' in event and 'authorization' in event['headers']:
        if event['headers']['authorization'] == DECRYPTED:
            response = {
                "isAuthorized": True,
            }
        else:
            response = {
                "isAuthorized": False,
            }
    else:
        response = {
            "isAuthorized": False,
            "message": "Missing headers or authorization key",
        }

    return response


#def lambda_handler(event, context):
#    ANON_KEY = os.environ.get('ANON_KEY')
#
#    # Ensure 'headers' key exists in the event
#    if 'headers' in event and 'authorization' in event['headers']:
#        if event['headers']['authorization'] == ANON_KEY:
#            response = {
#                "isAuthorized": True,
#            }
#        else:
#            response = {
#                "isAuthorized": False,
#            }
#    else:
#        response = {
#            "isAuthorized": False,
#            "message": "Missing headers or authorization key",
#        }
#
#    return response