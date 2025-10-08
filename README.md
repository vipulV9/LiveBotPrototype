 Step 1
GEMINI_API_KEY=your_gemini_api_key_here    google cloud(  https://cloud.google.com/)  → api & services  → credential → create a new api key  


 Step 2
REDIS_PASSWORD=your_redis_password_here    Go to 👉 https://redis.io/try-free/   →  create a db →  Copy the url which looks like this and in the code line num 54 change these with your url REDIS_HOST = "redis-12173.crce217.ap-south-1-1.ec2.redns.redis-cloud.com" 
REDIS_PORT = 11111
REDIS_USERNAME = "default"
 Step 3
 TO  GET GOOGLE_APPLICATION_CREDENTIALS=C:\\Users\\YourName\\Desktop\\gcloud_key.json


In the sidebar, go to IAM & Admin → Service Accounts

or directly: https://console.cloud.google.com/iam-admin/serviceaccounts
Click “Create Service Account”.
Enter:
Name: flask-tts-service
Role: Project → Editor (for testing; you can restrict later)

Click Done.

 Generate the JSON Key File
In your new service account list, click “Actions → Manage Keys”.
Go to the Keys tab.
Click “Add Key → Create new key”.
Choose JSON → Click Create.


 A JSON file will be downloaded automatically  || REMEMBER TO RENAME IT TO “gcloud_key.json” ||

This is your Google Credentials File.


REPLACE THE PATH FROM THE ENV FILE   
GOOGLE_APPLICATION_CREDENTIALS=C:\\Users\\YourName\\Desktop\\gcloud_key.json
GOOGLE_CLOUD_PROJECT=your proj name 
GOOGLE_CLOUD_REGION=us-central1


