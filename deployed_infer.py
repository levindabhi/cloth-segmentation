import argparse
import base64
import datetime
import io
import json
from google.api import httpbody_pb2
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from PIL import Image

parser = argparse.ArgumentParser(description='Make cloth segmentation predictions on an image using Google Cloud Vertex AI Platform.')

parser.add_argument('-i', '--input', type=str, required=True, help='path to the input image(jpg, jpeg or png) file')
parser.add_argument('-o', '--output', type=str, required=True, help='folder to the output PNG image file')
parser.add_argument('-p', '--project', type=str, required=True, help='google cloud project ID where it is to be deployed')
parser.add_argument('-l', '--location', type=str, required=True, help='location of the server')
parser.add_argument('-p_no', '--project_number', type=str, required=True, help='the project number on google cloud')
parser.add_argument('-e_id', '--endpoint_id', type=str, required=True, help='endpoint id on which the model is deployed')

args = parser.parse_args()

project = args.project
location = args.location

# initiating the aiplatform
aiplatform.init(project=project, location=location)

#processing the input image
input_img = Image.open(args.input)
img_bytes = io.BytesIO()
if (args.input[-3:].lower()=='jpg') or (args.input[-4:].lower()=='jpeg'): 
    input_img.save(img_bytes, format='JPEG')
elif args.input[-3:].lower()=='png':
    input_img.save(img_bytes, format='PNG')
output = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

instances = {"instances":[{"instance_key": output}]}

endpoint = aiplatform.Endpoint(f"projects/{args.project_number}/locations/{location}/endpoints/{args.endpoint_id}")

prediction_client = aiplatform_v1.PredictionServiceClient(
    client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
)

http_body = httpbody_pb2.HttpBody(
        data=json.dumps(instances).encode("utf-8"),
        content_type="application/json",
    )
  
ENDPOINT_RESOURCE_NAME = endpoint.resource_name

request = aiplatform_v1.RawPredictRequest(
    endpoint=ENDPOINT_RESOURCE_NAME,
    http_body=http_body,
)

# performing inference
check1 = prediction_client.raw_predict(request=request)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
body = check1.data[13:-34].decode()

# saving the output
Image.open(io.BytesIO(base64.b64decode(body))).save(args.output+f"output-{timestamp}.png", format='PNG')
print(f'Successfully saved the PNG image to {args.output}/output-{timestamp}.png')

