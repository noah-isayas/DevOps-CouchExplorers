import base64
import boto3
import json
import os
import random

def lambda_handler(event, context):
    # Her setter jeg opp AWS klienter
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    s3_client = boto3.client("s3")

    # Hent miljøvariabler fra AWS Lambda-konfigurasjonen
    model_id = os.environ.get("MODEL_ID", "amazon.titan-image-generator-v1")
    bucket_name = os.environ.get("BUCKET_NAME", "pgr301-couch-explorers")
    candidate_number = os.environ.get("CANDIDATE_NUMBER", "12345")  # Default value if not set

    # Vi må trenger prompten til bildet fra request body, denne tas ut her
    request_body = json.loads(event['body'])
    prompt = request_body.get("prompt", "Standard tekst hvis prompt ikke er oppgitt")

    # Genererer seed
    seed = random.randint(0, 2147483647)
    s3_image_path = f"{candidate_number}/generated_images/titan_{seed}.png"

    native_request = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": 8.0,
            "height": 1024,
            "width": 1024,
            "seed": seed,
        }
    }

    # Påkaller bedrock-modellen
    try:
        response = bedrock_client.invoke_model(modelId=model_id, body=json.dumps(native_request))
        model_response = json.loads(response["body"].read())

        # Henter ut og dekrypterer bilde data til base64
        base64_image_data = model_response["images"][0]
        image_data = base64.b64decode(base64_image_data)

        # Laster opp den dekrypterte bildedataen til S3
        s3_client.put_object(Bucket=bucket_name, Key=s3_image_path, Body=image_data)

        # Returnerer statuskode med filstien til bildet
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Image generated and uploaded successfully",
                "s3_image_path": f"s3://{bucket_name}/{s3_image_path}"
            })
        }
    except Exception as e:
        # Return error response if something goes wrong
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": "Failed to generate or upload image",
                "error": str(e)
            })
        }
